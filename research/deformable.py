"""Bounded local deformation for pedestrian tracks with sparse SMPL observations."""

import torch
from torch import nn

from .geometry import matrix_quat, quat_matrix


GAUSSIAN_PARAMETERS = ("means", "quats", "scales", "opacities", "features_dc", "features_rest")


class PedestrianGaussians(nn.Module):
    def __init__(self, actor_index, seed, times, present, size, deform, max_offset=0.20):
        super().__init__()
        self.actor_index = actor_index
        self.max_offset = max_offset
        for name in GAUSSIAN_PARAMETERS:
            self.register_parameter(name, nn.Parameter(seed[name].detach().clone()))
        self.register_buffer("times", times.detach().clone())
        self.register_buffer("present", present.detach().clone())
        if deform:
            # Spatial anchors let the two legs and the torso move differently.
            anchors = size.new_tensor([
                [0, 0, 0.36], [0, 0, 0.10], [-0.26, 0, -0.26],
                [0.26, 0, -0.26], [-0.36, 0, 0.13], [0.36, 0, 0.13],
            ])
            normalized = seed["means"].detach() / size.clamp_min(0.1)
            influence = (-torch.cdist(normalized, anchors).square() / 0.08).softmax(-1)
            # Global displacement already comes from the cuboid trajectory.
            centered = influence - influence.mean(0)
            self.register_buffer("influence", centered / centered.abs().sum(-1).max().clamp_min(1))
            self.offsets = nn.Parameter(seed["means"].new_zeros(len(times), len(anchors), 3))
        else:
            self.register_parameter("offsets", None)

    def time_span(self, time):
        hi = torch.searchsorted(self.times, time.reshape(())).clamp(max=len(self.times) - 1)
        # Use the preceding interval at a knot, including the last knot.
        lo = (hi - 1).clamp(min=0)
        dt = (self.times[hi] - self.times[lo]).clamp_min(1e-6)
        alpha = ((time - self.times[lo]) / dt).clamp(0, 1)
        valid = self.present[lo] & self.present[hi] & (self.times[hi] - self.times[lo] <= 0.20)
        nearest = torch.where(self.present[hi] & ((alpha >= 0.5) | ~self.present[lo]), hi, lo)
        return torch.where(valid, lo, nearest), torch.where(valid, hi, nearest), alpha, dt

    def get_gaussians_at(self, time, box, velocity, visible):
        local = self.means
        deform_velocity = torch.zeros_like(local)
        if self.offsets is not None:
            lo, hi, alpha, dt = self.time_span(time)
            start, end = self.offsets[lo].tanh(), self.offsets[hi].tanh()
            delta = self.influence @ start.lerp(end, alpha) * self.max_offset
            local = local + delta
            deform_velocity = (self.influence @ (end - start) * (self.max_offset / dt)).detach()
            deform_velocity = deform_velocity * (self.present[hi] & self.present[lo])
        rotation = box[:3, :3]
        linear, angular = velocity.split(3)
        rigid_velocity = linear + torch.cross(angular.expand_as(local), local, dim=-1) @ rotation.T
        return {
            "means": local @ rotation.T + box[:3, 3],
            "quats": matrix_quat(rotation @ quat_matrix(self.quats)),
            "scales": self.scales.exp(),
            "opacities": self.opacities.sigmoid().squeeze(-1) * visible,
            "features_dc": self.features_dc,
            "features_rest": self.features_rest,
            "velocities": rigid_velocity + deform_velocity @ rotation.T,
        }

    def regularization(self, time):
        lo, hi, alpha, _ = self.time_span(time)
        offsets = self.offsets.tanh() * self.max_offset
        at_time = offsets[lo].lerp(offsets[hi], alpha)
        valid = self.present[:-2] & self.present[1:-1] & self.present[2:]
        valid = valid & (self.times[2:] - self.times[:-2] <= 0.4)
        acceleration = offsets[2:] - 2 * offsets[1:-1] + offsets[:-2]
        smooth = (acceleration.abs().mean(dim=(1, 2)) * valid).sum() / valid.sum().clamp_min(1)
        return at_time.square().mean(), smooth
