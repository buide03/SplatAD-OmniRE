from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import torch
from torch import nn

from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig

from .geometry import BodyModel
from .prepare import read_json, sha256, validate_tracks
from .smpl_nodes import SMPLTrack

POSE_GROUPS = ("smpl_translation", "smpl_root_rotation", "smpl_body_rotation")


@dataclass
class PedestrianModelConfig(SplatADModelConfig):
    _target: Type = field(default_factory=lambda: PedestrianModel)
    variant: Literal["R0", "R1", "R2"] = "R0"
    scene_id: str = "011"
    tracks_path: Path | None = None
    manifest_path: Path | None = None
    body_path: Path | None = None
    manifest_digest: str = ""


class PedestrianModel(SplatADModel):
    refinement_enabled = False
    human_motion = True

    def split_seed_points(self, seed_points):
        background, actors = super().split_seed_points(seed_points)
        self.background_seeds = len(background)
        self.actor_seeds = [len(points) for points in actors]
        return background, actors

    def populate_modules(self):
        super().populate_modules()
        self.people = nn.ModuleList()
        self.person_index = {}
        self.last_sensor = None
        self.last_rgb_index = None
        self.last_actor_time = None
        cfg = self.config
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
        manifest = read_json(cfg.manifest_path)
        document = validate_tracks(read_json(cfg.tracks_path), cfg.scene_id)
        if sha256(cfg.tracks_path) != manifest["tracks_sha256"] or sha256(cfg.body_path) != document["body_sha256"]:
            raise ValueError("The pose JSON or neutral mesh does not match the frozen manifest")
        tracks = {track["uuid"]: track for track in document["tracks"]}
        selected = set(manifest["selected_uuids"])
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
                    cfg.feature_dim, cfg.init_opacities, cfg.variant == "R2",
                ))
            replaced.append(index)
            route = self.actor_routes[index]
            route["rigid_gaussians_replaced"] = route["gaussians"]
            route.update(representation="smpl", gaussians=len(body.vertices))
        if not replaced:
            raise ValueError("No selected UUID with at least two valid SMPL poses matched a PandaSet trajectory")
        if selected - self.person_index.keys():
            print("Selected UUIDs left rigid:", sorted(selected - self.person_index.keys()))
        # Cuboid seeding ran before removal, keeping these points out of the background.
        keep = ~torch.isin(self.id.flatten().long(), torch.tensor(replaced, dtype=torch.long))
        for name, parameter in self.gauss_params.items():
            self.gauss_params[name] = nn.Parameter(parameter.detach()[keep], requires_grad=parameter.requires_grad)

    def scene_gaussians(self, time):
        gaussians = super().scene_gaussians(time)
        if self.training:
            self.last_actor_time = time.detach()
        if not self.people:
            return gaussians
        articulated = [node.get_gaussians_at(time, self.human_motion) for node in self.people]
        return {name: torch.cat([value, *(person[name] for person in articulated)])
                for name, value in gaussians.items()}

    def step_post_backward(self, step):
        # Raster info now contains multiple parameter groups. Keep the point budget fixed.
        pass

    def load_state_dict(self, state_dict, **kwargs):
        # Fixed counts preserve the parameter objects and their optimizer contract.
        return nn.Module.load_state_dict(self, state_dict, **kwargs)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        if self.training:
            self.last_sensor = "rgb" if "image" in batch else "lidar"
            image_index = batch.get("image_idx") if self.last_sensor == "rgb" else None
            self.last_rgb_index = int(image_index) if image_index is not None else None
        return super().get_loss_dict(outputs, batch, metrics_dict)

    def get_param_groups(self):
        groups = super().get_param_groups()
        if self.people:
            for name in ("means", "quats", "scales", "opacities", "features_dc", "features_rest"):
                groups[f"smpl_{name}"] = [getattr(person, name) for person in self.people]
            if self.config.variant == "R2":
                for name, attr in zip(POSE_GROUPS, ("translation", "root", "body"), strict=True):
                    groups[name] = [getattr(person, attr) for person in self.people]
        return groups
