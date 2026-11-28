"""R3 far pedestrians: SMPL driven by a walking cycle borrowed from a near SMPL track of the scene.

4D-Humans cannot pose a person who is 30-90 px tall, and a free deformation MLP learns no walk from
such crops (photometric gradients cannot pull a leg several pixels across). A far walker therefore
gets the skeleton and skinning of SMPL, with the pose initialized from the scene's mean gait cycle:

* each near walker lends the lag of its track whose leg joints repeat best (0.6-1.6 s);
* the cycles are turned to walk along +x, resampled, aligned in phase and averaged;
* the phase advances with the distance the far cuboid travels, one cycle per stride (stride per
  metre of box height from the donors, cadence clamped to 0.5-1.5x theirs), so legs stop when the
  box stops; research.far_fit picks the starting phase that best explains the walker's 2D
  keypoints and LiDAR returns;
* the cycle is turned about the box's vertical axis to the walker's own direction, the pelvis is
  centred in the box and the stance foot stands on the box floor;
* the hip swing is scaled until the stance foot stays put at the box's speed: 4D-Humans and the
  averaging of several donors both damp the swing, leaving feet that slide with the box;
* the body is as tall as its box: the first SMPL shape coefficient is set to match the box height
  (clamped to 1.45-1.95 m).

Afterwards the pose trains like any other R3 SMPL track (per-frame refinement, smoothing, box), plus
the keypoint and LiDAR terms of research.far_fit.
"""
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .geometry import apply, blend, joint_transforms, matrix_quat, quat_matrix, slerp

LEG_ROWS = (0, 1, 3, 4)  # Body rows of the left/right hip and knee (SMPL joint index minus the pelvis).
MIN_PERIOD_S, MAX_PERIOD_S = 0.6, 1.6
MIN_SWING_DEG = 5.0  # A donor cycle whose legs move less than this is not a walk.
CADENCE_RANGE = (0.5, 1.5)
COLOUR_RADIUS_M = 0.15
HIP_ROWS = (0, 1)  # Body rows of the left/right hip.
ANKLES = (7, 8)  # SMPL joints of the left/right ankle.
LEG_SCALES = tuple(0.5 + 0.25 * k for k in range(15))  # 0.5x to 4x the donor hip swing.
MAX_HIP_SWING_DEG = 35.0  # Hip flexion/extension about its mean stays within a human walk.
STATURE_RANGE_M = (1.45, 1.95)  # Adult statures; a box outside this is more likely loose than the person.
SHAPE_LIMIT = 3.0  # SMPL shape coefficients are in standard deviations of its training bodies.


def stature(body, betas):
    """Head-to-sole extent of the rest mesh along the body's own up axis (ankles to head joint)."""
    shaped = body.vertices + body.shapes @ betas
    joints = body.regressor @ shaped
    up = F.normalize(joints[15] - (joints[7] + joints[8]) / 2, dim=0)
    height = shaped @ up
    return float(height.max() - height.min())


def stature_betas(body, height):
    """SMPL shape whose stature matches a far walker's box height, through the first shape component.

    That component mostly scales stature. Its value is found by bisection within +-3 standard deviations,
    so an implausible target ends at the limit instead of a distorted body. Bodies without shape blend
    shapes keep their template.
    """
    betas = torch.zeros(10)
    if getattr(body, "shapes", None) is None:
        return betas
    target = min(max(float(height), STATURE_RANGE_M[0]), STATURE_RANGE_M[1])
    at = lambda value: stature(body, torch.tensor([value] + [0.0] * 9))
    low, high = -SHAPE_LIMIT, SHAPE_LIMIT
    if at(low) > at(high):
        low, high = high, low
    if target <= at(low):
        betas[0] = low
    elif target >= at(high):
        betas[0] = high
    else:
        for _ in range(40):
            middle = (low + high) / 2
            low, high = (middle, high) if at(middle) < target else (low, middle)
        betas[0] = (low + high) / 2
    return betas


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
def gait_cycle(person, rotations, positions, height):
    """The best closed walking cycle of a box-anchored SMPL track, from its initial (4D-Humans) pose.

    ``rotations`` [F, 3, 3] and ``positions`` [F, 3] are the track's anchor box at its pose times and
    ``height`` its box height, which turns the stride into a stride per metre of height.
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
        "period_s": float(times[first + lag] - times[first]), "stride_per_height": stride / height,
        "direction": direction, "pelvis": person.joints[0].clone(),
        "closing_deg": round(math.degrees(closing), 2), "swing_deg": round(math.degrees(swing), 2), "score": score,
    }


def spin_about_z(angle, like):
    return quat_matrix(like.new_tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]))


def turned(root, translation, pelvis, angle):
    """Turn local poses about the box's vertical axis. Roots pivot about the pelvis, so the offset turns too."""
    spin = spin_about_z(angle, root)
    return matrix_quat(spin @ quat_matrix(root)), (pelvis + translation) @ spin.T - pelvis


def heading(direction):
    return math.atan2(float(direction[1]), float(direction[0]))


def mean_quat(stack, anchor):
    """Average of unit quaternions [N, ..., 4] after flipping each to the hemisphere of ``anchor``."""
    stack = torch.where((stack * anchor).sum(-1, keepdim=True) < 0, -stack, stack)
    return F.normalize(stack.mean(0), dim=-1)


@torch.no_grad()
def average_cycle(cycles, samples=24):
    """The scene's mean gait: cycles turned to walk along +x, resampled, aligned in phase, averaged.

    The cleanest cycle (lowest score) is the phase reference; each other cycle is rolled to the
    shift whose hip and knee rotations match it best.
    """
    phases = torch.arange(samples, dtype=torch.float32) / samples
    ordered = sorted(cycles, key=lambda c: c["score"])
    aligned = []
    for cycle in ordered:
        root, body, translation = sample_cycle(cycle, phases)
        root, translation = turned(root, translation, cycle["pelvis"], -heading(cycle["direction"]))
        if aligned:
            legs, target = body[:, LEG_ROWS], aligned[0][1][:, LEG_ROWS]
            shift = min(range(samples), key=lambda k: float(quat_angle(legs.roll(-k, 0), target).mean()))
            root, body, translation = root.roll(-shift, 0), body.roll(-shift, 0), translation.roll(-shift, 0)
        aligned.append((root, body, translation - translation.mean(0)))
    first = aligned[0]
    return {
        "uuid": "+".join(c["uuid"] for c in ordered), "donors": [c["uuid"] for c in ordered],
        "root": mean_quat(torch.stack([a[0] for a in aligned]), first[0]),
        "body": mean_quat(torch.stack([a[1] for a in aligned]), first[1]),
        "translation": torch.stack([a[2] for a in aligned]).mean(0),
        "period_s": sum(c["period_s"] for c in ordered) / len(ordered),
        "stride_per_height": sum(c["stride_per_height"] for c in ordered) / len(ordered),
        "direction": torch.tensor([1.0, 0.0]), "pelvis": ordered[0]["pelvis"],
        "closing_deg": max(c["closing_deg"] for c in ordered), "swing_deg": min(c["swing_deg"] for c in ordered),
    }


LIBRARY_FIELDS = ("root", "body", "translation", "pelvis", "direction")


def save_cycle(path, cycle, scene_id):
    """Keep a scene's mean gait for scenes without a near walker (tensors as lists)."""
    record = {key: value.tolist() if torch.is_tensor(value) else value for key, value in cycle.items()}
    record["scene_id"] = str(scene_id)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(record)):
        return False
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return True


def library_cycle(folder, scene_id):
    """The mean of the gait cycles other scenes saved in ``folder``, or None."""
    cycles = []
    for path in sorted(Path(folder).glob("*.json")) if folder is not None and Path(folder).is_dir() else []:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("scene_id") == str(scene_id):
            continue
        for key in LIBRARY_FIELDS:
            record[key] = torch.tensor(record[key], dtype=torch.float32)
        record["uuid"] = f"scene {record['scene_id']}"
        record.setdefault("score", 0.0)
        cycles.append(record)
    return average_cycle(cycles) if cycles else None


def floor_planes(person, height):
    """The annotated cuboid floor at each pose, as scene-frame planes (normal, offset) keyed by frame."""
    up = person.anchor_rotation[..., :, 2]
    point = person.anchor_position - height / 2 * up
    offset = -(up * point).sum(-1)
    return {str(int(f)): [*n.tolist(), float(o)] for f, n, o in zip(person.frame_ids, up, offset)}


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


def local_mesh(person, root, body, translation, points=None):
    """SMPL vertices (or ``points`` skinned like them) [T, V, 3] in the box frame, before the box transform."""
    rotations = quat_matrix(torch.cat((root[..., None, :], body), dim=-2))
    transform = joint_transforms(rotations, person.joints, person.parents) @ person.inverse_canonical
    transform = torch.cat((transform[..., :3, :3], transform[..., :3, 3:] + translation[..., None, :, None]), dim=-1)
    return apply(blend(person.weights, transform), person.vertices if points is None else points)


def local_joints(person, root, body, translation):
    """The 24 SMPL joints [T, 24, 3] in the box frame, before the box transform."""
    rotations = quat_matrix(torch.cat((root[..., None, :], body), dim=-2))
    return apply(joint_transforms(rotations, person.joints, person.parents)[..., :3, :], person.joints) + translation[..., None, :]


def foot_slip(person, root, body, translation, times, rotations, positions):
    """Median scene-frame speed (m/s) of the slower ankle between consecutive poses: the stance foot's slide."""
    ankles = local_joints(person, root, body, translation)[:, list(ANKLES)]
    world = ankles @ rotations.transpose(-1, -2) + positions[:, None]
    seconds = times[1:] - times[:-1]
    keep = (seconds > 0) & (seconds <= 0.2)
    if not keep.any():
        return 0.0
    speed = (world[1:] - world[:-1]).norm(dim=-1).amin(-1) / seconds.clamp_min(1e-6)
    return float(speed[keep].median())


def swing_hips(body, scale):
    """Hip rotations moved ``scale`` times as far from their cycle mean (1 keeps them)."""
    body = body.clone()
    for row in HIP_ROWS:
        mean = mean_quat(body[:, row], body[:1, row]).expand_as(body[:, row])
        body[:, row] = slerp(mean, body[:, row], body.new_tensor(float(scale)))
    return body


def hip_swing_deg(body):
    """Largest hip rotation away from its cycle mean, in degrees."""
    return max(math.degrees(float(quat_angle(mean_quat(body[:, row], body[:1, row]), body[:, row]).max()))
               for row in HIP_ROWS)


@torch.no_grad()
def gait_poses(cycle, person, times, rotations, positions, height, phase=0.0):
    """Local root, body and translation of a far walker at its box times, and what was decided.

    ``phase`` (in cycles) is where the walker is in its stride when its box starts moving.
    """
    stride = cycle["stride_per_height"] * height
    distance = path_length(positions)
    seconds = float(times[-1] - times[0])
    speed = float(distance[-1]) / seconds if seconds > 0 else 0.0
    if speed > 0:
        # Keep the step rate within a plausible range of the donor's, whatever the box speed.
        cadence = 1.0 / cycle["period_s"]
        low, high = speed / (CADENCE_RANGE[1] * cadence), speed / (CADENCE_RANGE[0] * cadence)
        stride = min(max(stride, low), high)
    root, body, translation = sample_cycle(cycle, phase + distance / stride)
    turn = 0.0
    direction = local_direction(rotations, positions)
    if direction is not None:
        turn = heading(direction) - heading(cycle["direction"])
    pelvis = person.joints[0]
    root, translation = turned(root, translation, pelvis, turn)
    translation[:, :2] -= (pelvis[:2] + translation[:, :2]).mean(0)
    slip = lambda legs: foot_slip(person, root, legs, translation, times, rotations, positions)
    before, swing = slip(body), hip_swing_deg(body)
    scales = [k for k in LEG_SCALES if k * swing <= MAX_HIP_SWING_DEG] or [1.0]
    scale = min(scales, key=lambda k: slip(swing_hips(body, k)))
    body = swing_hips(body, scale)
    floor = local_mesh(person, root, body, translation)[..., 2].amin(-1).median()
    translation[:, 2] += -height / 2 - floor
    decided = {"gait_phase": round(float(phase) % 1.0, 3), "gait_stride_m": round(stride, 3), "gait_cadence_hz": round(speed / stride, 3) if speed > 0 else 0.0,
               "gait_turn_deg": round(math.degrees(turn), 1), "gait_speed_mps": round(speed, 3),
               "gait_hip_scale": scale, "gait_hip_swing_deg": round(scale * swing, 1),
               "foot_slip_before_mps": round(before, 3), "foot_slip_mps": round(slip(body), 3)}
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
