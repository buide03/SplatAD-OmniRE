"""Canonical DA-pose, vertex-area Gaussian initialization and LBS adapted from OmniRe."""
import math

import numpy as np
import torch
from torch.nn import functional as F

JOINTS = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)


def quat_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def matrix_quat(r):
    """Use the best-conditioned component, including near 180°."""
    a, b, c, d, e, f, g, h, i = r.flatten(-2).unbind(-1)
    sq = torch.stack((1+a+e+i, 1+a-e-i, 1-a+e-i, 1-a-e+i), dim=-1)
    candidates = torch.stack((
        sq[..., 0], h-f, c-g, d-b, h-f, sq[..., 1], b+d, c+g,
        c-g, b+d, sq[..., 2], f+h, d-b, c+g, f+h, sq[..., 3],
    ), dim=-1).reshape(*r.shape[:-2], 4, 4)
    index = sq.argmax(dim=-1)[..., None, None].expand(*sq.shape[:-1], 1, 4)
    return F.normalize(candidates.gather(-2, index).squeeze(-2), dim=-1)


def slerp(a, b, alpha):
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs()
    angle = dot.clamp(max=0.9995).acos()
    curved = (((1 - alpha) * angle).sin() * a + (alpha * angle).sin() * b) / angle.sin()
    return F.normalize(torch.where(dot > 0.9995, a.lerp(b, alpha), curved), dim=-1)


def joint_transforms(rot, joints, parents):
    """SMPL rest-to-pose transforms, including rotation around the root joint."""
    relative = torch.cat((joints[:1], joints[1:] - joints[list(parents[1:])]))
    local = torch.cat((rot, relative[..., None].expand(*rot.shape[:-2], 3, 1)), dim=-1)
    bottom = rot.new_tensor([0, 0, 0, 1]).expand(*local.shape[:-2], 1, 4)
    local = torch.cat((local, bottom), dim=-2).unbind(-3)
    accumulated = [local[0]]
    for j, parent in enumerate(parents[1:], 1):
        accumulated.append(accumulated[parent] @ local[j])
    absolute = torch.stack(accumulated, dim=-3)
    rotation = absolute[..., :3, :3]
    shift = absolute[..., :3, 3:] - rotation @ joints[..., None]
    return torch.cat((torch.cat((rotation, shift), dim=-1), bottom), dim=-2)


def blend(weights, transforms):
    return torch.einsum("vj,...jrc->...vrc", weights, transforms[..., :3, :])


def apply(transform, points):
    return (transform[..., :3] @ points[..., None]).squeeze(-1) + transform[..., 3]


class BodyModel:
    """Read compact, non-pickle SMPL inputs exported by prepare-smpl."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as source:
            self.version = str(source["smpl_version"])
            self.vertices = torch.tensor(source["v_template"], dtype=torch.float32)
            self.shapes = torch.tensor(source["shapedirs"], dtype=torch.float32)[..., :10]
            self.poses = torch.tensor(source["posedirs"], dtype=torch.float32)
            self.regressor = torch.tensor(source["J_regressor"], dtype=torch.float32)
            self.weights = torch.tensor(source["weights"], dtype=torch.float32)
            self.faces = torch.tensor(source["f"].astype(np.int64))
            self.parents = tuple(int(p) for p in source["parents"])
        n = len(self.vertices)
        expected = ((self.vertices, (n, 3)), (self.shapes, (n, 3, 10)),
                    (self.poses, (n, 3, 207)), (self.regressor, (24, n)), (self.weights, (n, 24)))
        if any(x.shape != shape or not torch.isfinite(x).all() for x, shape in expected):
            raise ValueError("Invalid SMPL mesh or blend shapes")
        if (len(self.parents) != 24 or self.parents[0] != -1
                or any(not 0 <= p < j for j, p in enumerate(self.parents[1:], 1))
                or not torch.allclose(self.weights.sum(-1), torch.ones(n), atol=1e-4)):
            raise ValueError("Invalid SMPL joint tree or skinning weights")

    def canonical(self, betas):
        shaped = self.vertices + self.shapes @ betas
        joints = self.regressor @ shaped
        angle = torch.zeros(24)
        angle[1], angle[2] = math.pi / 6, -math.pi / 6
        q = torch.stack(((angle / 2).cos(), angle * 0, angle * 0, (angle / 2).sin()), -1)
        rot = quat_matrix(q)
        offsets = self.poses @ (rot[1:] - torch.eye(3)).flatten()
        canonical = joint_transforms(rot, joints, self.parents)
        vertices = apply(blend(self.weights, canonical), shaped + offsets)
        return vertices, joints, torch.linalg.inv(canonical)


def mesh_gaussians(vertices, faces):
    triangles = vertices[faces]
    normal = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1)
    area = normal.norm(dim=-1) / 2
    vertex_area = torch.zeros_like(vertices[:, 0])
    vertex_normals = torch.zeros_like(vertices)
    for corner in faces.unbind(-1):
        vertex_area.index_add_(0, corner, area / 3)
        vertex_normals.index_add_(0, corner, normal)
    normal = F.normalize(vertex_normals, dim=-1)
    fallback = normal.new_tensor([0, 0, 1]).expand_as(normal)
    normal = torch.where(normal.norm(dim=-1, keepdim=True) > 0, normal, fallback)
    axis = F.one_hot(normal.abs().argmin(-1), 3).to(normal)
    tangent = F.normalize(torch.cross(normal, axis, dim=-1), dim=-1)
    bitangent = torch.cross(normal, tangent, dim=-1)
    quat = matrix_quat(torch.stack((tangent, bitangent, normal), dim=-1))
    radius = (vertex_area / math.pi).sqrt().clamp(1e-4, 0.1)
    log_scale = torch.stack((radius, radius, (radius * 0.5).clamp(min=1e-4)), dim=-1).log()
    return quat, log_scale
