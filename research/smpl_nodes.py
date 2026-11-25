"""A fixed set of articulated Gaussians per PandaSet UUID."""
import torch
from torch import nn

from .geometry import apply, blend, joint_transforms, matrix_quat, quat_matrix, mesh_gaussians, slerp


class SMPLTrack(nn.Module):
    def __init__(self, track, body, scene_from_world, time_offset, feature_dim, opacity, optimize):
        super().__init__()
        self.uuid = track["uuid"]
        self.parents = body.parents
        betas = torch.tensor(track["betas"], dtype=torch.float32)
        vertices, joints, inverse = body.canonical(betas)
        frames = [frame for frame in track["frames"] if frame["valid"]]
        times = torch.tensor([f["timestamp_s"] - time_offset for f in frames], dtype=torch.float32)
        frame_ids = torch.tensor([f["frame_idx"] for f in frames])
        roots = torch.tensor([f["root_translation"] for f in frames], dtype=torch.float32)
        root_quats = torch.tensor([f["root_quat_wxyz"] for f in frames], dtype=torch.float32)
        body_quats = torch.tensor([f["body_quats_wxyz"] for f in frames], dtype=torch.float32)
        rotation = scene_from_world[:3, :3].float()
        # SMPL root rotations pivot about the pelvis, so rotate the offset about it too.
        roots = (roots + joints[0]) @ rotation.T + scene_from_world[:3, 3] - joints[0]
        root_quats = matrix_quat(rotation @ quat_matrix(root_quats))
        for name, x in {
            "betas": betas, "vertices": vertices, "joints": joints, "inverse_canonical": inverse,
            "faces": body.faces, "weights": body.weights, "times": times, "frame_ids": frame_ids,
            "vertex_ids": torch.arange(len(vertices)), "initial_translation": roots.clone(),
            "initial_root": root_quats.clone(), "initial_body": body_quats.clone(),
        }.items():
            self.register_buffer(name, x)
        self.translation = nn.Parameter(roots, requires_grad=optimize)
        self.root = nn.Parameter(root_quats, requires_grad=optimize)
        self.body = nn.Parameter(body_quats, requires_grad=optimize)
        quats, scales = mesh_gaussians(vertices, body.faces)
        self.means = nn.Parameter(vertices.clone())
        self.quats = nn.Parameter(quats)
        self.scales = nn.Parameter(scales)
        self.opacities = nn.Parameter(torch.full_like(vertices[:, :1], opacity).logit())
        self.features_dc = nn.Parameter(torch.full_like(vertices, 0.5))
        self.features_rest = nn.Parameter(torch.randn(len(vertices), feature_dim))

    def _span(self, time):
        """Query consecutive valid poses or briefly hold the endpoint of a segment."""
        index = torch.searchsorted(self.times, time)
        lo, hi = (index - 1).clamp(0, len(self.times) - 1), index.clamp(0, len(self.times) - 1)
        near = torch.where((time - self.times[lo]).abs() <= (time - self.times[hi]).abs(), lo, hi)
        contiguous = (self.frame_ids[hi] - self.frame_ids[lo] == 1) & (self.times[hi] - self.times[lo] <= 0.2)
        visible = contiguous | ((time - self.times[near]).abs() <= 0.055)
        lo, hi = torch.where(contiguous, lo, near), torch.where(contiguous, hi, near)
        alpha = ((time - self.times[lo]) / (self.times[hi] - self.times[lo]).clamp_min(1e-6)).clamp(0, 1)
        return lo, hi, alpha, visible

    def _world_transforms(self, index, other=None, alpha=None, initial=False):
        roots = self.initial_root if initial else self.root
        body = self.initial_body if initial else self.body
        translation = self.initial_translation if initial else self.translation
        q = torch.cat((roots[:, None], body), dim=1)
        pose, shift = q[index], translation[index]
        if other is not None:
            pose = slerp(pose, q[other], alpha)
            shift = shift.lerp(translation[other], alpha)
        transform = joint_transforms(quat_matrix(pose), self.joints, self.parents) @ self.inverse_canonical
        return torch.cat((transform[..., :3, :3], transform[..., :3, 3:] + shift[:, None]), dim=-1)

    def mesh_at(self, time, initial=False):
        lo, hi, alpha, visible = self._span(time.to(self.times).reshape(()))
        transform = self._world_transforms(lo, hi, alpha, initial)
        return apply(blend(self.weights, transform), self.vertices), visible

    def get_gaussians_at(self, time, motion=True):
        lo, hi, alpha, visible = self._span(time.to(self.times).reshape(()))
        transform = blend(self.weights, self._world_transforms(lo, hi, alpha))
        means = apply(transform, self.means)
        quats = matrix_quat(transform[..., :3] @ quat_matrix(self.quats))
        velocity = torch.zeros_like(means)
        if motion:
            with torch.no_grad():
                prev = (lo - 1).clamp_min(0)
                following = (hi + 1).clamp_max(len(self.times) - 1)
                same = lo == hi
                start = torch.where(same & (self.frame_ids[lo] - self.frame_ids[prev] == 1)
                                    & (self.times[lo] - self.times[prev] <= 0.2), prev, lo)
                end = torch.where(same & (self.frame_ids[following] - self.frame_ids[hi] == 1)
                                  & (self.times[following] - self.times[hi] <= 0.2), following, hi)
                dt = (self.times[end] - self.times[start]).clamp_min(1e-6)
                difference = self._world_transforms(end) - self._world_transforms(start)
                velocity = apply(blend(self.weights, difference / dt), self.means) * visible
        return {
            "means": means, "quats": quats, "velocities": velocity,
            "scales": self.scales.exp(), "opacities": self.opacities.sigmoid().squeeze(-1) * visible,
            "features_dc": self.features_dc, "features_rest": self.features_rest,
        }
