"""R3 deformable pedestrians: OmniRe's DeformableNodes inside SplatAD's actor cuboids."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import matrix_quat, quat_matrix


GAUSSIAN_PARAMETERS = ("means", "quats", "scales", "opacities", "features_dc", "features_rest")


def random_quaternions(count, generator=None):
    """Uniform random rotations, as SplatAD's random_quat_tensor."""
    u, v, w = torch.rand(3, count, generator=generator)
    return torch.stack(((1 - u).sqrt() * torch.sin(2 * torch.pi * v), (1 - u).sqrt() * torch.cos(2 * torch.pi * v),
                        u.sqrt() * torch.sin(2 * torch.pi * w), u.sqrt() * torch.cos(2 * torch.pi * w)), dim=-1)


def seed_gaussians(points, colors, opacity_logit, init_scale, feature_dim, generator=None):
    """Gaussian rows from box-frame points and RGB 0-255, with SplatAD's create_gauss_param_dict rules:
    scale from the mean distance to the 3 nearest points, random rotation, colour / 255 as features_dc."""
    means = torch.as_tensor(points, dtype=torch.float32)
    count = len(means)
    if count < 4:
        spacing = torch.ones(count, 1)
    else:
        spacing = torch.cat([torch.cdist(chunk, means).topk(4, largest=False).values[:, 1:].mean(-1, keepdim=True)
                             for chunk in means.split(1024)])
    return {
        "means": means,
        "quats": random_quaternions(count, generator),
        "scales": (spacing.clamp_min(1e-4) * init_scale).log().repeat(1, 3),
        "opacities": torch.full((count, 1), float(opacity_logit)),
        "features_dc": torch.as_tensor(colors, dtype=torch.float32) / 255,
        "features_rest": torch.randn(count, feature_dim, generator=generator),
    }


class OmniReDeformationField(nn.Module):
    """OmniRe's ConditionalDeformNetwork (DriveStudio, MIT): D=8, W=256, skip at D/2, 10 frequency bands."""

    embedding_dim = 16

    def __init__(self, depth=8, width=256, bands=10):
        super().__init__()
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


class DeformablePedestrian(nn.Module):
    """Gaussians of one moving non-SMPL pedestrian, expressed in its cuboid frame.

    Before the warm-up ends the node is rigid and its canonical Gaussians train as in SplatAD.
    Afterwards OmniRe's MLP predicts per-Gaussian offsets and the canonical positions stop training
    (OmniRe ``stop_optimizing_canonical_xyz``).
    """

    def __init__(self, actor_index, seed, scene_times, size):
        super().__init__()
        self.actor_index = actor_index
        for name in GAUSSIAN_PARAMETERS:
            self.register_parameter(name, nn.Parameter(seed[name].detach().clone()))
        self.register_buffer("scene_times", scene_times.detach().clone())
        self.register_buffer("size", size.detach().clone())
        self.embedding = nn.Parameter(seed["means"].new_empty(OmniReDeformationField.embedding_dim).uniform_())

    def normalized_time(self, time):
        start, end = self.scene_times[0], self.scene_times[-1]
        return ((time.to(self.scene_times).reshape(()) - start) / (end - start).clamp_min(1e-6)).clamp(0, 1)

    def get_gaussians_at(self, time, box, velocity, visible, field=None):
        """Return the SplatAD packet and the OmniRe out-of-bound penalty (None while rigid)."""
        local, quats = self.means, self.quats
        deform_velocity = torch.zeros_like(local)
        outside = None
        if field is not None:
            canonical = self.means.detach()
            centers = 2 * canonical / self.size[2].clamp_min(1e-6)
            t = self.normalized_time(time)
            if self.training and torch.is_grad_enabled():
                delta, rotation_delta = checkpoint(field, centers, t, self.embedding,
                                                   use_reentrant=False, preserve_rng_state=False)
            else:
                delta, rotation_delta = field(centers, t, self.embedding)
            local = canonical + delta
            quats = F.normalize(self.quats, dim=-1) + rotation_delta
            # SplatAD's rolling shutter and LiDAR sweep need velocities in m/s.
            with torch.no_grad():
                step = 0.005
                span = (self.scene_times[-1] - self.scene_times[0]).clamp_min(1e-6)
                direction = 1.0 if float(t) + step / float(span) <= 1 else -1.0
                shifted, _ = field(centers, t + direction * step / span, self.embedding)
                deform_velocity = (shifted - delta) * direction / step
            outside = (local.abs() - self.size / 2).relu().mean()
        rotation = box[:3, :3]
        linear, angular = velocity.split(3)
        rigid_velocity = linear + torch.cross(angular.expand_as(local), local, dim=-1) @ rotation.T
        packet = {
            "means": local @ rotation.T + box[:3, 3],
            "quats": matrix_quat(rotation @ quat_matrix(quats)),
            "scales": self.scales.exp(),
            "opacities": self.opacities.sigmoid().squeeze(-1) * visible,
            "features_dc": self.features_dc,
            "features_rest": self.features_rest,
            "velocities": rigid_velocity + deform_velocity @ rotation.T,
        }
        return packet, outside
