"""R3 far walkers: penalties from 2D keypoints and per-frame LiDAR, and the starting gait phase.

Photometric gradients cannot move a leg of a 30-90 px person several pixels, so the borrowed gait
(research.gait) is tied to two kinds of evidence whose gradient points at the answer from afar:

* keypoints: SMPL joints projected with the PandaSet camera of each ViTPose observation, against
  its 12 body keypoints (shoulders, elbows, wrists, hips, knees, ankles), weighted by confidence;
* LiDAR: each frame's Pandar64 returns inside the walker's cuboid (above its floor) against the
  posed Gaussian centres of that frame, both in the scene frame, so the term also reaches the box.

Both use the Geman-McClure penalty r^2 / (r^2 + sigma^2): quadratic near the answer, bounded for
outliers such as a swapped left/right keypoint or a return from an occluder. Before training, the
gait phase is chosen from a grid to minimise the same two terms.
"""
import torch
from torch import nn

from .gait import gait_poses, local_mesh

# COCO keypoint index -> SMPL joint index. The face has no SMPL joint.
COCO_TO_SMPL = ((5, 16), (6, 17), (7, 18), (8, 19), (9, 20), (10, 21),
                (11, 1), (12, 2), (13, 4), (14, 5), (15, 7), (16, 8))
KEYPOINT_SIGMA = 0.1  # In box heights.
LIDAR_SIGMA_M = 0.1
FLOOR_MARGIN_M = 0.05  # Returns this close to the box floor are road.
MATCH_S = 0.06  # A keypoint or sweep belongs to the pose closer than this in time.


def gmof(squared, sigma):
    return squared / (squared + sigma ** 2)


def match(times, query):
    """Nearest pose index for each query time, and whether it is close enough."""
    index = (times[None] - query[:, None]).abs().argmin(-1)
    return index, (times[index] - query).abs() <= MATCH_S


class FarFit(nn.Module):
    """Keypoint and LiDAR targets of one far walker, as non-persistent buffers that follow the model."""

    def __init__(self, person, keypoints, sweeps, scene_from_world, time_offset, lidar_times, height,
                 min_score=0.3):
        super().__init__()
        times = person.times
        self.register_buffer("scene_rotation", scene_from_world[:3, :3].float().clone(), persistent=False)
        self.register_buffer("scene_origin", scene_from_world[:3, 3].float().clone(), persistent=False)
        smpl = torch.tensor([j for _, j in COCO_TO_SMPL])
        self.register_buffer("smpl_joints", smpl, persistent=False)
        rows = []
        if keypoints:
            stamps = torch.tensor([r["timestamp_s"] - time_offset for r in keypoints], dtype=times.dtype)
            index, ok = match(times, stamps)
            for row, i, good in zip(keypoints, index.tolist(), ok.tolist(), strict=True):
                xy = torch.tensor(row["keypoints_xy"], dtype=torch.float32)[[c for c, _ in COCO_TO_SMPL]]
                score = torch.tensor(row["scores"], dtype=torch.float32)[[c for c, _ in COCO_TO_SMPL]]
                weight = torch.where(score >= min_score, score, torch.zeros_like(score))
                if good and weight.sum() > 0:
                    lens = row["intrinsics"]
                    x1, y1, x2, y2 = row["bbox_xyxy"]
                    rows.append((i, torch.tensor(row["camera_rotation"]), torch.tensor(row["camera_origin"]),
                                 torch.tensor([lens["fx"], lens["fy"]]), torch.tensor([lens["cx"], lens["cy"]]),
                                 xy, weight, max(y2 - y1, 1.0)))
        columns = list(zip(*rows)) if rows else [[]] * 8
        stack = lambda values, shape: torch.stack(values).float() if values else torch.zeros(shape)
        self.register_buffer("kp_index", torch.tensor(columns[0], dtype=torch.long), persistent=False)
        self.register_buffer("kp_rotation", stack(columns[1], (0, 3, 3)), persistent=False)
        self.register_buffer("kp_origin", stack(columns[2], (0, 3)), persistent=False)
        self.register_buffer("kp_focal", stack(columns[3], (0, 2)), persistent=False)
        self.register_buffer("kp_centre", stack(columns[4], (0, 2)), persistent=False)
        self.register_buffer("kp_xy", stack(columns[5], (0, len(COCO_TO_SMPL), 2)), persistent=False)
        self.register_buffer("kp_weight", stack(columns[6], (0, len(COCO_TO_SMPL))), persistent=False)
        self.register_buffer("kp_scale", torch.tensor(columns[7], dtype=torch.float32), persistent=False)

        groups, points, owners = [], [], []
        for frame, returns in sweeps or []:
            if frame >= len(lidar_times):
                continue
            i, good = match(times, times.new_tensor([lidar_times[frame] - time_offset]))
            returns = torch.as_tensor(returns, dtype=torch.float32)
            returns = returns[returns[:, 2] > -height / 2 + FLOOR_MARGIN_M]
            if bool(good) and len(returns):
                owners.append(torch.full((len(returns),), len(groups)))
                groups.append(int(i))
                points.append(returns)
        self.register_buffer("lidar_index", torch.tensor(groups, dtype=torch.long), persistent=False)
        self.register_buffer("lidar_points", torch.cat(points) if points else torch.zeros(0, 3), persistent=False)
        self.register_buffer("lidar_owner", torch.cat(owners) if owners else torch.zeros(0, dtype=torch.long),
                             persistent=False)

    @property
    def keypoint_observations(self):
        return len(self.kp_index)

    @property
    def lidar_frames(self):
        return len(self.lidar_index)

    def project(self, person):
        """Pixels [N, 12, 2] of the SMPL joints at each keypoint observation, and which are in front."""
        joints = person.joints_at(self.kp_index)[:, self.smpl_joints]
        world = (joints - self.scene_origin) @ self.scene_rotation
        local = torch.einsum("njc,ncd->njd", world - self.kp_origin[:, None], self.kp_rotation)
        depth = local[..., 2]
        pixel = local[..., :2] / depth.clamp_min(0.5)[..., None] * self.kp_focal[:, None] + self.kp_centre[:, None]
        return pixel, depth > 0.5

    def keypoint_loss(self, person):
        if not self.keypoint_observations:
            return None
        pixel, front = self.project(person)
        squared = (pixel - self.kp_xy).square().sum(-1) / self.kp_scale[:, None].square()
        weight = self.kp_weight * front
        return (gmof(squared, KEYPOINT_SIGMA) * weight).sum() / weight.sum().clamp_min(1e-6)

    def lidar_loss(self, person, frames=None):
        """Mean penalty of the returns of ``frames`` random sweeps (all when None) to the posed Gaussians."""
        if not self.lidar_frames:
            return None
        chosen = torch.arange(self.lidar_frames, device=self.lidar_index.device)
        if frames is not None and self.lidar_frames > frames:
            chosen = torch.randperm(self.lidar_frames, device=self.lidar_index.device)[:frames]
        index = self.lidar_index[chosen]
        mesh = local_mesh(person, person.root[index], person.body[index], person.translation[index], person.means)
        rotation, position = person.box_at(person.times[index])
        mesh = mesh @ rotation.transpose(-1, -2) + position[:, None]
        anchor_rotation, anchor_position = person.anchor_rotation[index], person.anchor_position[index]
        penalties = []
        for slot, group in enumerate(chosen.tolist()):
            returns = self.lidar_points[self.lidar_owner == group]
            # The returns were cut with the annotated cuboid, which is the anchor box of this pose.
            returns = returns @ anchor_rotation[slot].T + anchor_position[slot]
            nearest = torch.cdist(returns, mesh[slot]).min(-1).values
            penalties.append(gmof(nearest.square(), LIDAR_SIGMA_M))
        return torch.cat(penalties).mean()

    @torch.no_grad()
    def summary(self, person):
        result = {"keypoint_observations": self.keypoint_observations, "lidar_frames": self.lidar_frames}
        if self.keypoint_observations:
            pixel, front = self.project(person)
            error = (pixel - self.kp_xy).norm(dim=-1)[(self.kp_weight > 0) & front]
            result["keypoint_error_px"] = round(float(error.median()), 2) if len(error) else None
        if self.lidar_frames:
            result["lidar_fit"] = round(float(self.lidar_loss(person)), 4)
        return result

    @torch.no_grad()
    def cost(self, person):
        terms = [term for term in (self.keypoint_loss(person), self.lidar_loss(person)) if term is not None]
        return float(sum(terms)) if terms else 0.0


@torch.no_grad()
def choose_phase(person, fit, cycle, times, rotations, centers, height, steps):
    """Pose ``person`` at the grid phase that best explains its keypoints and LiDAR; return what was decided."""
    best = None
    for k in range(steps if fit.keypoint_observations or fit.lidar_frames else 1):
        root, body, shift, decided = gait_poses(cycle, person, times, rotations, centers, height, phase=k / steps)
        for name, value in (("root", root), ("body", body), ("translation", shift)):
            getattr(person, name).copy_(value)
        cost = fit.cost(person)
        if best is None or cost < best[0]:
            best = (cost, (root, body, shift), decided)
    cost, poses, decided = best
    for name, value in zip(("root", "body", "translation"), poses, strict=True):
        getattr(person, name).copy_(value)
        getattr(person, f"initial_{name}").copy_(value)
    return {**decided, "gait_phase_cost": round(cost, 4), **fit.summary(person)}
