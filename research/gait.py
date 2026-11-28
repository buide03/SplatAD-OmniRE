"""R3 far pedestrians: SMPL driven by a walking cycle borrowed from a near SMPL track of the scene.

4D-Humans cannot pose a person who is 30-90 px tall, and a free deformation MLP learns no walk from
such crops (photometric gradients cannot pull a leg several pixels across). A far walker therefore
gets the skeleton and skinning of SMPL, with the pose initialized from one closed gait cycle of the
best-observed near walker:

* the cycle is the lag of the near track whose leg joints repeat best (0.6-1.6 s);
* the phase advances with the distance the far cuboid travels, one cycle per stride (stride scaled
  by box height, cadence clamped to 0.5-1.5x the donor's), so legs stop when the box stops;
* the cycle is turned about the box's vertical axis so that both people walk the same way in their
  own boxes, the pelvis is centred in the box and the stance foot stands on the box floor.

Afterwards the pose trains like any other R3 SMPL track (per-frame refinement, smoothing, box).
"""
import math

import torch
from torch.nn import functional as F

from .geometry import apply, blend, joint_transforms, matrix_quat, quat_matrix, slerp

LEG_ROWS = (0, 1, 3, 4)  # Body rows of the left/right hip and knee (SMPL joint index minus the pelvis).
MIN_PERIOD_S, MAX_PERIOD_S = 0.6, 1.6
MIN_SWING_DEG = 5.0  # A donor cycle whose legs move less than this is not a walk.
CADENCE_RANGE = (0.5, 1.5)
COLOUR_RADIUS_M = 0.15


def quat_angle(a, b):
    dot = (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1).abs().clamp(max=1.0)
    return 2 * dot.acos()


def contiguous_runs(frame_ids, times):
    gaps = ((frame_ids[1:] - frame_ids[:-1] != 1) | (times[1:] - times[:-1] > 0.2)).nonzero().flatten() + 1
    bounds = [0, *gaps.tolist(), len(frame_ids)]
    return list(zip(bounds[:-1], bounds[1:]))


def local_direction(rotations, positions):
    """Mean walking direction in the box's own frame (unit xy), or None when the box hardly moves."""
    steps = positions[1:] - positions[:-1]
    local = (steps[..., None, :] @ rotations[:-1]).squeeze(-2)[:, :2].sum(0)
    return F.normalize(local, dim=0) if local.norm() > 0.05 else None


def path_length(positions, window=5):
    """Cumulative horizontal distance of a lightly smoothed box centre [T].

    The ends are padded by point reflection, so steady walking keeps its full length.
    """
    xy = positions[:, :2]
    half = window // 2
    if len(xy) > half:
        before = 2 * xy[:1] - xy[1:half + 1].flip(0)
        after = 2 * xy[-1:] - xy[-half - 1:-1].flip(0)
        xy = torch.cat((before, xy, after)).unfold(0, window, 1).mean(-1)
    steps = (xy[1:] - xy[:-1]).norm(dim=-1)
    return torch.cat((steps.new_zeros(1), steps.cumsum(0)))


@torch.no_grad()
def gait_cycle(person, rotations, positions):
    """The best closed walking cycle of a box-anchored SMPL track, from its initial (4D-Humans) pose.

    ``rotations`` [F, 3, 3] and ``positions`` [F, 3] are the track's anchor box at its pose times.
    Returns None when no contiguous run holds a leg cycle between MIN_PERIOD_S and MAX_PERIOD_S.
    """
    times, legs = person.times, person.initial_body[:, LEG_ROWS]
    best = None
    for start, end in contiguous_runs(person.frame_ids, times):
        count = end - start
        if count < 4:
            continue
        dt = float(times[end - 1] - times[start]) / (count - 1)
        for lag in range(max(2, math.ceil(MIN_PERIOD_S / dt)), min(count - 3, math.floor(MAX_PERIOD_S / dt)) + 1):
            # A true period repeats at every start, not only where two poses happen to coincide.
            closing = quat_angle(legs[start:end - lag], legs[start + lag:end]).mean(-1)
            first = start + int(closing.argmin())
            cycle = legs[first:first + lag]
            swing = quat_angle(cycle, cycle[:1]).mean(-1).max()
            if math.degrees(float(swing)) < MIN_SWING_DEG:
                continue
            score = float(closing.mean() / swing)
            if best is None or score < best[0]:
                best = (score, first, lag, float(closing[first - start]), float(swing))
    if best is None:
        return None
    score, first, lag, closing, swing = best
    span = slice(first, first + lag)
    travelled = path_length(positions)
    stride = float(travelled[first + lag] - travelled[first])
    direction = local_direction(rotations[first:first + lag + 1], positions[first:first + lag + 1])
    if stride < 0.2 or direction is None:
        return None
    # Keep the pelvis bob of the cycle, not the drift of the person inside its box, so the cycle closes.
    shift = person.initial_translation[first:first + lag + 1]
    ramp = torch.arange(lag, dtype=shift.dtype)[:, None] / lag
    translation = shift[:-1] - ramp * (shift[-1] - shift[0])
    return {
        "uuid": person.uuid, "root": person.initial_root[span].clone(), "body": person.initial_body[span].clone(),
        "translation": translation.clone(),
        "period_s": float(times[first + lag] - times[first]), "stride_m": stride, "direction": direction,
        "closing_deg": round(math.degrees(closing), 2), "swing_deg": round(math.degrees(swing), 2), "score": score,
    }


def sample_cycle(cycle, phase):
    """Root, body and translation at continuous phases [T] (1.0 is one full cycle)."""
    count = len(cycle["root"])
    position = (phase % 1.0) * count
    lo = position.floor().long() % count
    hi = (lo + 1) % count
    alpha = (position - position.floor())[:, None]
    root = slerp(cycle["root"][lo], cycle["root"][hi], alpha)
    body = slerp(cycle["body"][lo], cycle["body"][hi], alpha[:, None])
    translation = cycle["translation"][lo].lerp(cycle["translation"][hi], alpha)
    return root, body, translation


def local_mesh(person, root, body, translation):
    """SMPL vertices [T, V, 3] in the box frame for local poses, as SMPLTrack before the box transform."""
    rotations = quat_matrix(torch.cat((root[..., None, :], body), dim=-2))
    transform = joint_transforms(rotations, person.joints, person.parents) @ person.inverse_canonical
    transform = torch.cat((transform[..., :3, :3], transform[..., :3, 3:] + translation[..., None, :, None]), dim=-1)
    return apply(blend(person.weights, transform), person.vertices)


@torch.no_grad()
def gait_poses(cycle, person, times, rotations, positions, height, donor_height):
    """Local root, body and translation of a far walker at its box times, and what was decided."""
    stride = cycle["stride_m"] * height / donor_height
    distance = path_length(positions)
    seconds = float(times[-1] - times[0])
    speed = float(distance[-1]) / seconds if seconds > 0 else 0.0
    if speed > 0:
        # Keep the step rate within a plausible range of the donor's, whatever the box speed.
        cadence = 1.0 / cycle["period_s"]
        low, high = speed / (CADENCE_RANGE[1] * cadence), speed / (CADENCE_RANGE[0] * cadence)
        stride = min(max(stride, low), high)
    root, body, translation = sample_cycle(cycle, distance / stride)
    turn = 0.0
    direction = local_direction(rotations, positions)
    if direction is not None:
        turn = math.atan2(float(direction[1]), float(direction[0])) - math.atan2(
            float(cycle["direction"][1]), float(cycle["direction"][0]))
    spin = quat_matrix(root.new_tensor([math.cos(turn / 2), 0.0, 0.0, math.sin(turn / 2)]))
    root = matrix_quat(spin @ quat_matrix(root))
    pelvis = person.joints[0]
    # Root rotations pivot about the pelvis, so turning the body turns its offset about the box axis too.
    translation = (pelvis + translation) @ spin.T - pelvis
    translation[:, :2] -= (pelvis[:2] + translation[:, :2]).mean(0)
    floor = local_mesh(person, root, body, translation)[..., 2].amin(-1).median()
    translation[:, 2] += -height / 2 - floor
    decided = {"gait_stride_m": round(stride, 3), "gait_cadence_hz": round(speed / stride, 3) if speed > 0 else 0.0,
               "gait_turn_deg": round(math.degrees(turn), 1), "gait_speed_mps": round(speed, 3)}
    return root, body, translation, decided


@torch.no_grad()
def paint(person, points, colours, frame):
    """Colour each SMPL Gaussian from the nearest dataset point in the box frame (RGB 0-255)."""
    points = torch.as_tensor(points, dtype=torch.float32)
    colours = torch.as_tensor(colours, dtype=torch.float32) / 255
    mesh = local_mesh(person, person.root[frame], person.body[frame], person.translation[frame])
    closest = [torch.cdist(chunk, points).min(-1) for chunk in mesh.split(1024)]
    distance = torch.cat([found.values for found in closest])[:, None]
    nearest = torch.cat([found.indices for found in closest])
    painted = torch.where(distance <= COLOUR_RADIUS_M, colours[nearest], colours.median(0).values)
    person.features_dc.copy_(painted)
    return float((distance <= COLOUR_RADIUS_M).float().mean())
