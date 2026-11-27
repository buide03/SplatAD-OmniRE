from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import torch
from torch import nn

from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig

from .deformable import GAUSSIAN_PARAMETERS, OmniReDeformationField, OmniRePedestrianGaussians, PedestrianGaussians
from .geometry import BodyModel
from .pose_constraints import PoseConstraints
from .prepare import PEDESTRIANS, read_json, sha256, validate_tracks
from .smpl_nodes import SMPLTrack

POSE_GROUPS = ("smpl_translation", "smpl_root_rotation", "smpl_body_rotation")
SMPL_VARIANTS = ("R1", "R2", "R3")
TRAINABLE_POSE_VARIANTS = ("R2", "R3")


@dataclass
class PedestrianModelConfig(SplatADModelConfig):
    _target: Type = field(default_factory=lambda: PedestrianModel)
    variant: Literal["R0", "R1", "R2", "R3"] = "R0"
    scene_id: str = "011"
    tracks_path: Path | None = None
    manifest_path: Path | None = None
    body_path: Path | None = None
    manifest_digest: str = ""
    smpl_init_opacity: float = 0.99
    pedestrian_mcmc: bool = False
    far_deformation: bool = False
    far_deformation_backend: Literal["anchors", "omnire"] = "anchors"
    far_deform_start: int = 3000
    far_max_gaussians: int = 6000
    ground_path: Path | None = None
    ground_digest: str = ""
    ground_prefit: bool = False
    joint_smooth_weight: float = 0.02
    contact_weight: float = 0.1
    slip_weight: float = 0.1
    person_rgb_weight: float = 0.15


class PedestrianModel(SplatADModel):
    refinement_enabled = False
    human_motion = True

    def split_seed_points(self, seed_points):
        background, actors = super().split_seed_points(seed_points)
        self.background_seeds = len(background)
        self.actor_seeds = [len(points) for points in actors]
        return background, actors

    def populate_modules(self):
        cfg = self.config
        self.refinement_enabled = cfg.variant == "R3" and cfg.pedestrian_mcmc
        super().populate_modules()
        self.people = nn.ModuleList()
        self.other_people = nn.ModuleList()
        self.far_deformation_field = None
        self.far_bound_losses = []
        self.person_regions = {}
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
        if cfg.variant == "R0":
            return
        if not 0.0 < cfg.smpl_init_opacity < 1.0:
            raise ValueError("SMPL initial opacity must be between 0 and 1")
        manifest = read_json(cfg.manifest_path)
        document = validate_tracks(read_json(cfg.tracks_path), cfg.scene_id)
        if sha256(cfg.tracks_path) != manifest["tracks_sha256"] or sha256(cfg.body_path) != document["body_sha256"]:
            raise ValueError("The pose JSON or neutral mesh does not match the frozen manifest")
        tracks = {track["uuid"]: track for track in document["tracks"]}
        selected = set(manifest["selected_uuids"])
        near_frames, far_frames = {}, {}
        planes = {}
        if cfg.variant == "R3":
            for row in manifest["observations"]:
                if row["usable"] and row["bin"] == "near" and row["smpl_valid"]:
                    near_frames.setdefault(row["uuid"], set()).add(row["frame"])
                if row["usable"] and row["bin"] == "far" and not row["stationary"]:
                    far_frames.setdefault(row["uuid"], set()).add(row["frame"])
            if cfg.ground_path:
                if sha256(cfg.ground_path) != cfg.ground_digest:
                    raise ValueError("The ground-plane input changed after the run was configured")
                ground = read_json(cfg.ground_path)
                if (ground["scene_id"] != cfg.scene_id or ground["manifest_sha256"] != cfg.manifest_digest
                        or ground["schema_version"] != 1):
                    raise ValueError("Ground planes must be generated from this run's manifest")
                planes = ground["planes"]
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
            self.person_index[uuid] = len(self.people)
            # Keep the starting LiDAR shuffle identical across rigid and SMPL runs.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(torch.initial_seed() + index)
                self.people.append(SMPLTrack(
                    track, body, metadata["scene_from_world"], metadata["scene_time_offset"],
                    feature_dim=cfg.feature_dim,
                    opacity=cfg.smpl_init_opacity,
                    optimize=cfg.variant in TRAINABLE_POSE_VARIANTS,
                ))
            if cfg.variant == "R3" and uuid in near_frames:
                person = self.people[-1]
                person.constraints = PoseConstraints(person, near_frames[uuid], planes.get(uuid, {}),
                                                     metadata["scene_from_world"], cfg.ground_prefit)
                self.actor_routes[index].update(ground_planes=int(person.constraints.valid_planes.sum()),
                                               stance_frames=int(person.constraints.contact.any(dim=-1).sum()))
            replaced.append(index)
            route = self.actor_routes[index]
            route["rigid_gaussians_replaced"] = route["gaussians"]
            route.update(representation="smpl", gaussians=len(body.vertices))
        if cfg.variant == "R3":
            replaced.extend(self.initialize_other_people(trajectories, set(replaced), far_frames))
        if not self.people and not self.other_people:
            raise ValueError("No selected UUID with at least two valid SMPL poses matched a PandaSet trajectory")
        if selected - self.person_index.keys():
            print("Selected UUIDs without an SMPL actor:", sorted(selected - self.person_index.keys()))
        # Cuboid seeding ran before removal, keeping these points out of the background.
        keep = ~torch.isin(self.id.flatten().long(), torch.tensor(replaced, device=self.id.device, dtype=torch.long))
        for name, parameter in self.gauss_params.items():
            self.gauss_params[name] = nn.Parameter(parameter.detach()[keep], requires_grad=parameter.requires_grad)

    def initialize_other_people(self, trajectories, replaced, far_frames):
        cfg = self.config
        ids = self.id.flatten().long()
        dynamic = self.dynamic_actors
        moved = []
        for index, trajectory in enumerate(trajectories):
            if index in replaced or trajectory["label"] not in PEDESTRIANS:
                continue
            deform = cfg.far_deformation and len(far_frames.get(trajectory["uuid"], ())) >= 4
            if not (deform or self.refinement_enabled):
                continue
            omnire = deform and cfg.far_deformation_backend == "omnire"
            rows = (ids == index).nonzero(as_tuple=True)[0]
            if deform and len(rows) > cfg.far_max_gaussians:
                generator = torch.Generator(device=rows.device).manual_seed(torch.initial_seed() + index)
                rows = rows[torch.randperm(len(rows), generator=generator, device=rows.device)[:cfg.far_max_gaussians]]
            if not len(rows):
                continue
            seed = {key: self.gauss_params[key][rows] for key in GAUSSIAN_PARAMETERS}
            node_args = (index, seed, dynamic.unique_timestamps, dynamic.actor_present_at_time[:, index],
                         dynamic.actor_sizes[index])
            if omnire:
                if self.far_deformation_field is None:
                    self.far_deformation_field = OmniReDeformationField().to(seed["means"])
                node = OmniRePedestrianGaussians(*node_args)
            else:
                node = PedestrianGaussians(*node_args, deform)
            self.other_people.append(node)
            route = self.actor_routes[index]
            route.update(representation="deformable" if deform else "rigid", gaussians=len(rows))
            if deform:
                route["deformation_backend"] = cfg.far_deformation_backend
            moved.append(index)
        return moved

    def scene_gaussians(self, time):
        gaussians = super().scene_gaussians(time)
        if self.training:
            self.last_actor_time = time.detach()
            self.far_bound_losses = []
        articulated = [node.get_gaussians_at(time, self.human_motion) for node in self.people]
        if self.other_people:
            boxes, present = self.dynamic_actors.get_boxes2world(time, flatten=False)
            boxes, present = boxes.reshape(-1, 4, 4), present.reshape(-1)
            velocities = self.dynamic_actors.get_velocities(time).reshape(-1, 6)
            field = self.far_deformation_field
            visible = present.tolist() if field is not None else None
            for node in self.other_people:
                index = node.actor_index
                args = (time, boxes[index], velocities[index], present[index])
                if isinstance(node, OmniRePedestrianGaussians):
                    if not visible[index]:
                        continue
                    if self.step >= self.config.far_deform_start:
                        packet, outside = node.deformed_gaussians_at(*args, field)
                        articulated.append(packet)
                        if self.training:
                            self.far_bound_losses.append(outside)
                        continue
                articulated.append(node.get_gaussians_at(*args))
        if not articulated:
            return gaussians
        return {name: torch.cat([value, *(person[name] for person in articulated)])
                for name, value in gaussians.items()}

    def step_post_backward(self, step):
        if self.refinement_enabled:
            # ADMCMC operates on the Gaussian parameter bank, independent of raster indices.
            super().step_post_backward(step)

    def update_to_step(self, step):
        self.step = max(0, step)

    def load_state_dict(self, state_dict, **kwargs):
        if self.refinement_enabled and "gauss_params.means" in state_dict:
            for name, parameter in self.gauss_params.items():
                saved = state_dict[f"gauss_params.{name}"]
                if saved.shape != parameter.shape:
                    self.gauss_params[name] = nn.Parameter(torch.empty_like(saved, device=parameter.device),
                                                           requires_grad=parameter.requires_grad)
        return nn.Module.load_state_dict(self, state_dict, **kwargs)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        if self.training:
            self.last_sensor = "rgb" if "image" in batch else "lidar"
            image_index = batch.get("image_idx") if self.last_sensor == "rgb" else None
            self.last_rgb_index = int(image_index) if image_index is not None else None
        losses = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training and self.config.variant == "R3":
            weights = {"joint_smooth": self.config.joint_smooth_weight, "foot_contact": self.config.contact_weight,
                       "ground_penetration": self.config.contact_weight, "foot_slip": self.config.slip_weight}
            terms = [node.constraints.losses(node, self.last_actor_time) for node in self.people
                     if hasattr(node, "constraints")]
            if terms:
                losses.update({name: torch.stack([term[name] for term in terms]).mean() * weight
                               for name, weight in weights.items() if weight > 0})
            deformation = [node.regularization(self.last_actor_time) for node in self.other_people if node.offsets is not None]
            if deformation:
                magnitude, smooth = torch.stack([torch.stack(pair) for pair in deformation]).mean(0)
                losses.update(deformation_magnitude=0.05 * magnitude, deformation_smooth=0.02 * smooth)
            if self.far_bound_losses:
                losses["deformation_out_of_bound"] = torch.stack(self.far_bound_losses).mean()
                self.far_bound_losses = []
            boxes = self.person_regions.get(self.last_rgb_index, ())
            if boxes and self.config.person_rgb_weight:
                target = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
                pred = outputs["rgb"]
                height, width = pred.shape[:2]
                error = (target[:height, :width] - pred).abs()
                factor = self._get_downscale_factor()
                boxes = [(x1 // factor, y1 // factor, (x2 + factor - 1) // factor, (y2 + factor - 1) // factor)
                         for x1, y1, x2, y2 in boxes]
                crops = [error[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes if x1 < width and y1 < height]
                if crops:
                    losses["person_bbox_rgb"] = torch.stack([crop.mean() for crop in crops]).mean() * self.config.person_rgb_weight
        return losses

    def get_param_groups(self):
        groups = super().get_param_groups()
        if self.people:
            for name in GAUSSIAN_PARAMETERS:
                groups[f"smpl_{name}"] = [getattr(person, name) for person in self.people]
            if self.config.variant in TRAINABLE_POSE_VARIANTS:
                for name, attr in zip(POSE_GROUPS, ("translation", "root", "body"), strict=True):
                    groups[name] = [getattr(person, attr) for person in self.people]
        if self.other_people:
            for name in GAUSSIAN_PARAMETERS:
                groups[f"pedestrian_{name}"] = [getattr(person, name) for person in self.other_people]
            offsets = [person.offsets for person in self.other_people if person.offsets is not None]
            if offsets:
                groups["pedestrian_deformation"] = offsets
        if self.far_deformation_field is not None:
            groups["pedestrian_deformation_network"] = list(self.far_deformation_field.parameters())
            groups["pedestrian_deformation_embedding"] = [person.embedding for person in self.other_people
                                                          if isinstance(person, OmniRePedestrianGaussians)]
        return groups
