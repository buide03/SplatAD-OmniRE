from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import torch
from torch import nn

from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig

from .deformable import GAUSSIAN_PARAMETERS, DeformablePedestrian, OmniReDeformationField
from .geometry import BodyModel
from .pose_constraints import PoseConstraints
from .prepare import PEDESTRIANS, read_json, sha256, validate_tracks
from .smpl_nodes import SMPLTrack
from .smpl_regularization import SMPLRegularizer

POSE_GROUPS = ("smpl_translation", "smpl_root_rotation", "smpl_body_rotation")
VARIANTS = ("R2", "R3")
SMPL_REGULARIZERS = ("trans_smooth", "joint_smooth", "local_offset", "knn", "max_scale", "sharp_shape")
CONTACT_WEIGHTS = {"foot_contact": "contact_weight", "ground_penetration": "contact_weight", "foot_slip": "slip_weight"}


@dataclass
class PedestrianModelConfig(SplatADModelConfig):
    _target: Type = field(default_factory=lambda: PedestrianModel)
    variant: Literal["R2", "R3"] = "R2"
    scene_id: str = "011"
    tracks_path: Path | None = None
    manifest_path: Path | None = None
    body_path: Path | None = None
    manifest_digest: str = ""
    smpl_init_opacity: float = 0.99
    # R3 only. Defaults follow DriveStudio configs/omnire.yaml.
    deform_start: int = 3000
    """Steps before OmniRe's deformation MLP starts on moving non-SMPL pedestrians (use_deformgs_after)."""
    deform_max_gaussians: int = 5000
    """Gaussians kept per deformable pedestrian (instance_max_pts)."""
    deform_min_travel: float = 0.5
    """Minimum summed cuboid travel in meters for a deformable pedestrian (traj_length_thres)."""
    smpl_follow_box: bool = True
    """Express each SMPL root in its learned cuboid frame, so trajectory optimization moves the mesh."""
    smpl_min_frames: int = 5
    """Frames with a direct 4D-Humans pose outside the far bin that a track needs to stay SMPL."""
    smpl_min_box_px: float = 80.0
    """Minimum bbox height for such a frame. Tracks below either limit become deformable pedestrians."""
    ground_path: Path | None = None
    ground_digest: str = ""
    ground_prefit: bool = True
    trans_smooth_weight: float = 0.01
    joint_smooth_weight: float = 0.01
    knn_weight: float = 0.001
    max_scale_weight: float = 0.05
    sharp_shape_weight: float = 1.0
    out_of_bound_weight: float = 1.0
    contact_weight: float = 0.1
    slip_weight: float = 0.1
    local_offset_weight: float = 0.01
    """Tether of the root to its start inside the learned box, in meters, for anchored tracks."""


class PedestrianModel(SplatADModel):
    """R2: SMPL pedestrians with trainable pose on SplatAD's own Gaussian bank.

    Background and vehicles train exactly as in SplatAD, including MCMC and actor flipping, so the
    variants differ only in how pedestrians are initialized and moved. R3 adds OmniRe deformable
    nodes for moving pedestrians without SMPL, OmniRe's SMPL regularizers, LiDAR ground contact
    and SMPL anchored to the learned box.
    """

    human_motion = True

    def split_seed_points(self, seed_points):
        background, actors = super().split_seed_points(seed_points)
        self.background_seeds = len(background)
        self.actor_seeds = [len(points) for points in actors]
        return background, actors

    def populate_modules(self):
        cfg = self.config
        r3 = cfg.variant == "R3"
        super().populate_modules()
        self.people = nn.ModuleList()
        self.deformable_people = nn.ModuleList()
        self.deformation_field = None
        self.deform_bounds = []
        self.smpl_regularizers = []
        self.person_index = {}
        self.last_sensor = None
        self.last_rgb_index = None
        self.last_actor_time = None
        metadata = self.kwargs["metadata"]
        trajectories = metadata["trajectories"]
        self.actor_routes = [
            {"uuid": t["uuid"], "label": t["label"], "representation": "rigid",
             "seed_points": self.actor_seeds[i], "gaussians": int((self.id == i).sum())}
            for i, t in enumerate(trajectories)
        ]
        self.background_gaussians = int((self.id == len(trajectories)).sum())
        if cfg.manifest_path is not None and sha256(cfg.manifest_path) != cfg.manifest_digest:
            raise ValueError("The frozen observation manifest changed after the run was configured")
        if not 0.0 < cfg.smpl_init_opacity < 1.0:
            raise ValueError("SMPL initial opacity must be between 0 and 1")
        manifest = read_json(cfg.manifest_path)
        document = validate_tracks(read_json(cfg.tracks_path), cfg.scene_id)
        if sha256(cfg.tracks_path) != manifest["tracks_sha256"] or sha256(cfg.body_path) != document["body_sha256"]:
            raise ValueError("The pose JSON or neutral mesh does not match the frozen manifest")
        tracks = {track["uuid"]: track for track in document["tracks"]}
        selected = set(manifest["selected_uuids"])
        near_frames, planes = self.ground_inputs(manifest) if r3 else ({}, {})
        quality = self.smpl_quality(manifest) if r3 else {}
        body = BodyModel(cfg.body_path)
        replaced = []
        for index, trajectory in enumerate(trajectories):
            uuid = trajectory["uuid"]
            if uuid not in selected or not trajectory["deformable"]:
                continue
            track = tracks[uuid]
            valid = [f for f in track["frames"] if f["valid"]]
            if len(valid) < 2:
                continue
            if r3 and quality.get(uuid, 0) < cfg.smpl_min_frames:
                # A far or poorly observed pose would drag the mesh; OmniRe's deformable node takes it.
                self.actor_routes[index]["smpl_rejected"] = f"{quality.get(uuid, 0)} close direct frames"
                continue
            self.person_index[uuid] = len(self.people)
            # Keep the starting LiDAR shuffle identical across rigid and SMPL runs.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(torch.initial_seed() + index)
                self.people.append(SMPLTrack(
                    track, body, metadata["scene_from_world"], metadata["scene_time_offset"],
                    feature_dim=cfg.feature_dim,
                    opacity=cfg.smpl_init_opacity,
                    optimize=True,
                ))
            replaced.append(index)
            route = self.actor_routes[index]
            route["rigid_gaussians_replaced"] = route["gaussians"]
            route.update(representation="smpl", gaussians=len(body.vertices))
            if r3:
                self.people[-1].exact_velocity = True
                if cfg.smpl_follow_box:
                    self.people[-1].anchor_to(self.box_reader(index))
                    route["follows_box"] = True
            if r3 and uuid in near_frames:
                person = self.people[-1]
                person.constraints = PoseConstraints(person, near_frames[uuid], planes.get(uuid, {}),
                                                     metadata["scene_from_world"], cfg.ground_prefit)
                route.update(ground_planes=int(person.constraints.valid_planes.sum()),
                             stance_frames=int(person.constraints.contact.any(dim=-1).sum()))
        if not replaced and not r3:
            raise ValueError("No selected UUID with at least two valid SMPL poses matched a PandaSet trajectory")
        if selected - self.person_index.keys():
            print("Selected UUIDs left rigid:", sorted(selected - self.person_index.keys()))
        if r3:
            for person in self.people:
                if person.box_at is not None:
                    # After the ground prefit, so the tether keeps that correction. Saved for resume.
                    person.register_buffer("tether_translation", person.translation.detach().clone())
            self.smpl_regularizers = [SMPLRegularizer(person) for person in self.people]
            replaced.extend(self.initialize_deformable_people(trajectories, set(replaced)))
            if not self.people and not self.deformable_people:
                raise ValueError("R3 found neither a usable SMPL track nor a moving pedestrian")
        # Cuboid seeding ran before removal, keeping these points out of the background.
        keep = ~torch.isin(self.id.flatten().long(), torch.tensor(replaced, device=self.id.device, dtype=torch.long))
        for name, parameter in self.gauss_params.items():
            self.gauss_params[name] = nn.Parameter(parameter.detach()[keep], requires_grad=parameter.requires_grad)

    def smpl_quality(self, manifest):
        """Per UUID, frames with a direct 4D-Humans pose, outside the far bin, with a tall enough box."""
        cfg = self.config
        frames = {}
        for row in manifest["observations"]:
            if (row.get("smpl_valid") and row.get("pose_source") == "4d_humans" and row.get("bin") != "far"
                    and row.get("bbox_height_px", 0) >= cfg.smpl_min_box_px):
                frames.setdefault(row["uuid"], set()).add(row["frame"])
        return {uuid: len(found) for uuid, found in frames.items()}

    def box_reader(self, index):
        """Rotations [T, 3, 3] and centers [T, 3] of one actor box, from the trajectory being optimized."""
        def box_at(times):
            times = times.reshape(-1)
            boxes, _ = self.dynamic_actors.get_boxes2world(times, flatten=False)
            box = boxes.reshape(len(times), -1, 4, 4)[:, index]
            return box[..., :3, :3], box[..., :3, 3]
        return box_at

    def ground_inputs(self, manifest):
        cfg = self.config
        near_frames = {}
        for row in manifest["observations"]:
            if row["usable"] and row["bin"] == "near" and row["smpl_valid"]:
                near_frames.setdefault(row["uuid"], set()).add(row["frame"])
        if cfg.ground_path is None:
            return near_frames, {}
        if sha256(cfg.ground_path) != cfg.ground_digest:
            raise ValueError("The ground-plane input changed after the run was configured")
        ground = read_json(cfg.ground_path)
        if (ground["schema_version"] != 1 or ground["scene_id"] != cfg.scene_id
                or ground["manifest_sha256"] != cfg.manifest_digest):
            raise ValueError("Ground planes must be generated from this run's manifest")
        return near_frames, ground["planes"]

    def initialize_deformable_people(self, trajectories, replaced):
        """OmniRe DeformableNodes: moving pedestrians without a usable SMPL track."""
        cfg = self.config
        ids = self.id.flatten().long()
        moved = []
        for index, trajectory in enumerate(trajectories):
            if index in replaced or trajectory["label"] not in PEDESTRIANS or trajectory["stationary"]:
                continue
            positions = trajectory["poses"][:, :3, 3]
            travel = float((positions[1:] - positions[:-1]).norm(dim=-1).sum())
            if travel < cfg.deform_min_travel:
                continue
            rows = (ids == index).nonzero(as_tuple=True)[0]
            if not len(rows):
                continue
            if len(rows) > cfg.deform_max_gaussians:
                generator = torch.Generator(device=rows.device).manual_seed(torch.initial_seed() + index)
                order = torch.randperm(len(rows), generator=generator, device=rows.device)
                rows = rows[order[:cfg.deform_max_gaussians]]
            seed = {name: self.gauss_params[name][rows] for name in GAUSSIAN_PARAMETERS}
            if self.deformation_field is None:
                self.deformation_field = OmniReDeformationField().to(seed["means"])
            self.deformable_people.append(DeformablePedestrian(
                index, seed, self.dynamic_actors.unique_timestamps, self.dynamic_actors.actor_sizes[index]))
            self.actor_routes[index].update(representation="deformable", gaussians=len(rows),
                                            travel_m=round(travel, 2))
            moved.append(index)
        return moved

    def scene_gaussians(self, time):
        gaussians = super().scene_gaussians(time)
        if self.training:
            self.last_actor_time = time.detach()
            self.deform_bounds = []
        if not self.people and not self.deformable_people:
            return gaussians
        articulated = [node.get_gaussians_at(time, self.human_motion) for node in self.people]
        if self.deformable_people:
            boxes, present = self.dynamic_actors.get_boxes2world(time, flatten=False)
            boxes, present = boxes.reshape(-1, 4, 4), present.reshape(-1)
            velocities = self.dynamic_actors.get_velocities(time).reshape(-1, 6)
            field = self.deformation_field if self.step >= self.config.deform_start else None
            for node in self.deformable_people:
                index = node.actor_index
                if not bool(present[index]):
                    continue
                packet, outside = node.get_gaussians_at(time, boxes[index], velocities[index], present[index], field)
                articulated.append(packet)
                if outside is not None and self.training:
                    self.deform_bounds.append(outside)
        return {name: torch.cat([value, *(person[name] for person in articulated)])
                for name, value in gaussians.items()}

    def update_to_step(self, step):
        super().update_to_step(step)
        if self.config.variant == "R3":
            # Evaluation and replay must know whether the deformation MLP was active at this checkpoint.
            self.step = max(0, step)

    # SplatAD's step_post_backward runs unchanged: gsplat's MCMC only edits gauss_params (background
    # and rigid actors), never the SMPL or deformable Gaussians appended after them.

    def load_state_dict(self, state_dict, **kwargs):
        # MCMC resizes the bank between checkpoints. Optimizers are rebuilt after loading.
        for name, parameter in self.gauss_params.items():
            saved = state_dict[f"gauss_params.{name}"]
            if saved.shape != parameter.shape:
                self.gauss_params[name] = nn.Parameter(torch.empty_like(saved, device=parameter.device),
                                                       requires_grad=parameter.requires_grad)
        # Fixed SMPL counts preserve the parameter objects and their optimizer contract.
        return nn.Module.load_state_dict(self, state_dict, **kwargs)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        if self.training:
            self.last_sensor = "rgb" if "image" in batch else "lidar"
            image_index = batch.get("image_idx") if self.last_sensor == "rgb" else None
            self.last_rgb_index = int(image_index) if image_index is not None else None
        losses = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training and self.config.variant == "R3":
            losses.update(self.r3_losses())
        return losses

    def r3_losses(self):
        cfg = self.config
        weights = {name: getattr(cfg, f"{name}_weight") for name in SMPL_REGULARIZERS}
        weights.update({name: getattr(cfg, attr) for name, attr in CONTACT_WEIGHTS.items()})
        terms = defaultdict(list)
        time = self.last_actor_time
        if time is not None:
            for person, regularizer in zip(self.people, self.smpl_regularizers):
                for name, value in regularizer.losses(person, time, self.step, weights).items():
                    terms[name].append(value)
                if hasattr(person, "constraints"):
                    for name, value in person.constraints.losses(person, time).items():
                        terms[name].append(value)
        losses = {f"r3_{name}": torch.stack(values).mean() * weights[name]
                  for name, values in terms.items() if weights[name]}
        if self.deform_bounds and cfg.out_of_bound_weight:
            losses["r3_out_of_bound"] = torch.stack(self.deform_bounds).mean() * cfg.out_of_bound_weight
        self.deform_bounds = []
        return losses

    def get_param_groups(self):
        groups = super().get_param_groups()
        if self.people:
            for name in GAUSSIAN_PARAMETERS:
                groups[f"smpl_{name}"] = [getattr(person, name) for person in self.people]
            for name, attr in zip(POSE_GROUPS, ("translation", "root", "body"), strict=True):
                groups[name] = [getattr(person, attr) for person in self.people]
        if self.deformable_people:
            for name in GAUSSIAN_PARAMETERS:
                groups[f"deformable_{name}"] = [getattr(person, name) for person in self.deformable_people]
            groups["deformable_embedding"] = [person.embedding for person in self.deformable_people]
            groups["deformable_network"] = list(self.deformation_field.parameters())
        return groups
