"""Local pedestrian deformation and conversion to SplatAD Gaussian inputs."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

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
        return self.scene_packet(local, self.quats, deform_velocity, box, velocity, visible)

    def scene_packet(self, local, quats, deform_velocity, box, velocity, visible):
        rotation = box[:3, :3]
        linear, angular = velocity.split(3)
        rigid_velocity = linear + torch.cross(angular.expand_as(local), local, dim=-1) @ rotation.T
        return {
            "means": local @ rotation.T + box[:3, 3],
            "quats": matrix_quat(rotation @ quat_matrix(quats)),
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


class OmniReDeformationField(nn.Module):
    """OmniRe's ConditionalDeformNetwork adapted from DriveStudio (MIT)."""

    embedding_dim = 16

    def __init__(self):
        super().__init__()
        bands, depth, width = 10, 8, 256
        self.register_buffer("frequencies", 2.0 ** torch.arange(bands), persistent=False)
        self.skip = depth // 2
        input_dim = 4 * (1 + 2 * bands) + self.embedding_dim
        self.layers = nn.ModuleList(
            nn.Linear(input_dim if i == 0 else width + (input_dim if i == self.skip + 1 else 0), width)
            for i in range(depth)
        )
        self.position_delta = nn.Linear(width, 3)
        self.rotation_delta = nn.Linear(width, 4)

    def encode(self, values):
        angles = values.unsqueeze(-2) * self.frequencies[:, None]
        waves = torch.stack((angles.sin(), angles.cos()), dim=-2).flatten(start_dim=-3)
        return torch.cat((values, waves), dim=-1)

    def forward(self, centers, time, embedding):
        time_features = self.encode(time.reshape(1, 1)).expand(len(centers), -1)
        inputs = torch.cat((self.encode(centers), time_features, embedding.expand(len(centers), -1)), dim=-1)
        hidden = inputs
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden).relu()
            if i == self.skip:
                hidden = torch.cat((inputs, hidden), dim=-1)
        return self.position_delta(hidden), self.rotation_delta(hidden)


class OmniRePedestrianGaussians(PedestrianGaussians):
    def __init__(self, actor_index, seed, times, present, size):
        super().__init__(actor_index, seed, times, present, size, deform=False)
        self.register_buffer("size", size.detach().clone())
        self.embedding = nn.Parameter(seed["means"].new_empty(OmniReDeformationField.embedding_dim).uniform_())

    def deformed_gaussians_at(self, time, box, velocity, visible, field):
        duration = (self.times[-1] - self.times[0]).clamp_min(1e-6)
        time = time.to(self.times).reshape(()).clamp(self.times[0], self.times[-1])
        normalized_time = (time - self.times[0]) / duration
        canonical = self.means.detach()
        centers = 2 * canonical / self.size[2].clamp_min(1e-6)
        if self.training and torch.is_grad_enabled():
            delta, rotation_delta = checkpoint(field, centers, normalized_time, self.embedding,
                                               use_reentrant=False, preserve_rng_state=False)
        else:
            delta, rotation_delta = field(centers, normalized_time, self.embedding)
        local = canonical + delta
        quats = F.normalize(self.quats, dim=-1) + rotation_delta

        # SplatAD uses velocities in meters per second for rolling shutter and LiDAR.
        with torch.no_grad():
            direction = torch.where(self.times[-1] - time >= 0.005, 1.0, -1.0)
            shifted = (time + direction * 0.005).clamp(self.times[0], self.times[-1])
            shifted_delta, _ = field(centers, (shifted - self.times[0]) / duration, self.embedding)
            deform_velocity = (shifted_delta - delta) * direction / (shifted - time).abs().clamp_min(1e-6)

        outside = (local.abs() - self.size / 2).relu().mean()
        return self.scene_packet(local, quats, deform_velocity, box, velocity, visible), outside
