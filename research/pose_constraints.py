"""R3 ground contact for near-range SMPL poses: stance feet on the local LiDAR road plane, no slip."""

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import apply, blend


class PoseConstraints(nn.Module):
    def __init__(self, person, near_frames, plane_by_frame, scene_from_world, prefit=False):
        super().__init__()
        rotation, origin = scene_from_world[:3, :3].float(), scene_from_world[:3, 3].float()
        frames = person.frame_ids.tolist()
        planes = person.vertices.new_tensor([plane_by_frame.get(str(f), [0, 0, 0, 0]) for f in frames])
        normals = planes[:, :3] @ rotation.T
        offsets = planes[:, 3] - normals @ origin
        near = torch.tensor([f in near_frames for f in frames], dtype=torch.bool)
        self.register_buffer("near", near)
        self.register_buffer("normals", normals)
        self.register_buffer("offsets", offsets)
        self.register_buffer("valid_planes", normals.norm(dim=-1) > 0.9)
        # Candidate vertices of the left/right foot and ankle, in canonical space.
        support = torch.stack((person.weights[:, [7, 10]].sum(-1), person.weights[:, [8, 11]].sum(-1)))
        candidates = int((support > 0.25).sum(-1).min())
        if candidates == 0:
            raise ValueError("SMPL mesh has no identifiable foot vertices")
        self.register_buffer("sole", support.topk(min(48, candidates)).indices)
        self.register_buffer("adjacent", (person.frame_ids[1:] - person.frame_ids[:-1] == 1)
                             & (person.times[1:] - person.times[:-1] <= 0.20))
        with torch.no_grad():
            idx = torch.arange(len(frames))
            feet, indices = self.feet_at(person, idx)
            heights = (feet * normals[:, None]).sum(-1) + offsets[:, None]
            interval = (person.times[1:] - person.times[:-1]).clamp_min(1e-3)
            speeds = (feet[1:] - feet[:-1]).norm(dim=-1) / interval[:, None]
            speeds = torch.where(self.adjacent[:, None], speeds, torch.full_like(speeds, torch.inf))
            end = torch.full_like(heights[:1], torch.inf)
            speed = torch.minimum(torch.cat((end, speeds)), torch.cat((speeds, end)))
            contact = self.valid_planes[:, None] & (heights.abs() <= 0.20) & (speed <= 0.65)
            self.register_buffer("contact", contact)
            self.register_buffer("contact_vertices", indices)
            if prefit:
                # Make a bounded vertical correction only for inferred stance frames.
                up = rotation[:, 2]
                ground_delta = -(heights * contact).sum(-1) / contact.sum(-1).clamp_min(1)
                vertical = ground_delta / (normals @ up).clamp_min(0.7)
                person.translation.add_(person.scene_to_local(vertical.clamp(-0.12, 0.12)[:, None] * up))

    def feet_at(self, person, frame):
        vertices = self.sole.flatten()
        world = apply(blend(person.weights[vertices], person._world_transforms(frame)), person.vertices[vertices])
        world = world.reshape(*frame.shape, *self.sole.shape, 3)
        heights = (world * self.normals[frame][..., None, None, :]).sum(-1)
        lowest = heights.topk(min(4, heights.shape[-1]), largest=False).indices
        points = world.gather(-2, lowest[..., None].expand(*lowest.shape, 3))
        indices = self.sole.expand(*frame.shape, *self.sole.shape).gather(-1, lowest)
        return points.mean(-2), indices

    def losses(self, person, time):
        frame = (person.times - time.reshape(())).abs().argmin()
        left = (frame - 1).clamp_min(0)
        near = self.near[frame] & ((person.times[frame] - time.reshape(())).abs() <= 0.10)
        feet, _ = self.feet_at(person, frame)
        normal, offset = self.normals[frame], self.offsets[frame]
        height = feet @ normal + offset
        stance = self.contact[frame] & near
        clearance = (F.smooth_l1_loss(height, torch.zeros_like(height), beta=0.04, reduction="none") * stance).sum() / 2
        penetration = (-height - 0.02).clamp_min(0).mean() * (near & self.valid_planes[frame])

        # Compare the same support vertices in consecutive frames while a foot is planted.
        indices = self.contact_vertices[frame].flatten()
        paired = torch.stack((left, frame))
        support = apply(blend(person.weights[indices], person._world_transforms(paired)), person.vertices[indices])
        motion = (support[1] - support[0]).reshape(2, -1, 3).mean(1)
        tangent = motion - (motion @ normal)[:, None] * normal
        dt = person.times[frame] - person.times[left]
        hold = stance & self.contact[left] & (person.frame_ids[frame] - person.frame_ids[left] == 1) & (dt <= 0.20)
        slip = (tangent.norm(dim=-1) * hold).sum() / 2
        return {"foot_contact": clearance, "ground_penetration": penetration, "foot_slip": slip}

    @torch.no_grad()
    def measure(self, person, time):
        frame = (person.times - time.reshape(())).abs().argmin()
        if not bool(self.valid_planes[frame]):
            return {}
        feet, _ = self.feet_at(person, frame)
        height = (feet @ self.normals[frame] + self.offsets[frame]) * 100
        return {"left_foot_height_cm": float(height[0]), "right_foot_height_cm": float(height[1]),
                "left_foot_stance": bool(self.contact[frame, 0]), "right_foot_stance": bool(self.contact[frame, 1])}
