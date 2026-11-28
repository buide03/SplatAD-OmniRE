from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import torch
from torch import nn

from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig

from .far_fit import FarFit, choose_phase
from .gait import average_cycle, floor_planes, gait_cycle, library_cycle, paint, save_cycle, stature_betas
from .geometry import BodyModel
from .pose_constraints import PoseConstraints
from .prepare import PEDESTRIANS, read_json, sha256, validate_tracks
from .smpl_nodes import GAUSSIAN_PARAMETERS, SMPLTrack
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
    # R3 only. Regularizer defaults follow DriveStudio configs/omnire.yaml.
    gait_min_travel: float = 0.5
    """Minimum summed cuboid travel in meters for a far walker (OmniRe traj_length_thres)."""
    gait_min_speed: float = 0.2
    """Minimum mean cuboid speed (m/s) for a far walker. Slower boxes are a standing person whose
    annotation jitters; a walk cycle would shuffle its legs, so it stays a rigid actor as in R2."""
    deform_seed_path: Path | None = None
    """Box-frame LiDAR returns and fill points coloured by the cameras, written by research.deform_seed.
    The aggregate colours the SMPL Gaussians of far walkers; per-frame returns feed the LiDAR term."""
    deform_seed_digest: str = ""
    deform_seed_min_points: int = 64
    """A far walker with fewer dataset points keeps the default grey SMPL colour."""
    keypoints_path: Path | None = None
    """ViTPose keypoints of the manifest observations, written by research.keypoints."""
    keypoints_digest: str = ""
    keypoint_min_score: float = 0.3
    keypoint_weight: float = 0.05
    """Far walkers: Geman-McClure penalty of projected SMPL joints against 2D keypoints (box heights)."""
    lidar_fit_weight: float = 0.05
    """Far walkers: Geman-McClure penalty of each frame's LiDAR returns to the posed Gaussians (meters)."""
    lidar_fit_frames: int = 4
    """Sweeps per far walker drawn at each step for the LiDAR term."""
    gait_phase_steps: int = 24
    """Grid of starting gait phases searched against the keypoints and LiDAR before training."""
    gait_library: Path | None = None
    """Folder of scene mean gaits. A scene with near walkers saves its own; one without borrows them."""
    far_ground: bool = True
    """Far walkers: stance feet on the annotated cuboid floor, no penetration, no slip."""
    smpl_follow_box: bool = True
    """Express each SMPL root in its learned cuboid frame, so trajectory optimization moves the mesh."""
    smpl_min_frames: int = 5
    """Frames with a direct 4D-Humans pose outside the far bin that a track needs to stay SMPL."""
    smpl_min_box_px: float = 80.0
    """Minimum bbox height for such a frame. Tracks below either limit walk with a borrowed gait cycle."""
    ground_path: Path | None = None
    ground_digest: str = ""
    ground_prefit: bool = True
    trans_smooth_weight: float = 0.01
    joint_smooth_weight: float = 0.01
    knn_weight: float = 0.001
    max_scale_weight: float = 0.05
    sharp_shape_weight: float = 1.0
    contact_weight: float = 0.1
    slip_weight: float = 0.1
    local_offset_weight: float = 0.01
    """Tether of the root to its start inside the learned box, in meters, for anchored tracks."""


class PedestrianModel(SplatADModel):
    """R2: SMPL pedestrians with trainable pose on SplatAD's own Gaussian bank.

    Background and vehicles train exactly as in SplatAD, including MCMC and actor flipping, so the
    variants differ only in how pedestrians are initialized and moved. R3 anchors SMPL to the learned
    box, adds OmniRe's SMPL regularizers and LiDAR ground contact, and gives walking pedestrians
    without a usable 4D-Humans pose an SMPL body driven by the scene's mean gait cycle, tied to their
    2D keypoints and per-frame LiDAR returns.
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
        self.far_fits = nn.ModuleDict()
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
                # A far or poorly observed pose would drag the mesh; the borrowed gait takes it.
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
        if r3:
            replaced.extend(self.initialize_gait_people(trajectories, set(replaced), body))
            if not self.people:
                raise ValueError("R3 found no usable SMPL track and no walker to give a borrowed gait")
            for person in self.people:
                if person.box_at is not None:
                    # After the ground prefit, so the tether keeps that correction. Saved for resume.
                    person.register_buffer("tether_translation", person.translation.detach().clone())
            self.smpl_regularizers = [SMPLRegularizer(person) for person in self.people]
        if selected - self.person_index.keys():
            print("Selected UUIDs left rigid:", sorted(selected - self.person_index.keys()))
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

    def deform_seeds(self):
        cfg = self.config
        if cfg.deform_seed_path is None:
            return {}
        if sha256(cfg.deform_seed_path) != cfg.deform_seed_digest:
            raise ValueError("The far-pedestrian seed file changed after the run was configured")
        from .deform_seed import load_seed

        return load_seed(cfg.deform_seed_path, cfg.scene_id)

    def gait_cycle(self, trajectories):
        """The scene's mean gait from its near SMPL walkers, else the gait library of other scenes, or None."""
        cfg = self.config
        index_of = {t["uuid"]: i for i, t in enumerate(trajectories)}
        cycles = []
        for person in self.people:
            if person.box_at is None:
                continue
            height = float(self.dynamic_actors.actor_sizes[index_of[person.uuid]][2])
            cycle = gait_cycle(person, person.anchor_rotation, person.anchor_position, height)
            if cycle is not None:
                cycles.append(cycle)
        if cycles:
            mean = average_cycle(cycles)
            mean["source"] = "scene"
            if cfg.gait_library is not None:
                save_cycle(Path(cfg.gait_library) / f"{cfg.scene_id}.json", mean, cfg.scene_id)
            return mean
        borrowed = library_cycle(cfg.gait_library, cfg.scene_id)
        if borrowed is not None:
            borrowed["source"] = "library"
        return borrowed

    def far_evidence(self):
        """ViTPose keypoints by UUID and (LiDAR timestamps, per-frame returns by UUID), when configured."""
        cfg = self.config
        keypoints, sweeps, lidar_times = {}, {}, []
        if cfg.keypoints_path is not None:
            if sha256(cfg.keypoints_path) != cfg.keypoints_digest:
                raise ValueError("The keypoint file changed after the run was configured")
            from .keypoints import load_keypoints

            for row in load_keypoints(cfg.keypoints_path, cfg.scene_id, cfg.manifest_digest):
                keypoints.setdefault(row["uuid"], []).append(row)
        if cfg.deform_seed_path is not None:
            from .deform_seed import load_lidar_frames

            lidar_times, sweeps = load_lidar_frames(cfg.deform_seed_path, cfg.scene_id)
        return keypoints, sweeps, lidar_times

    def initialize_gait_people(self, trajectories, replaced, body):
        """Walking pedestrians without a usable 4D-Humans pose: SMPL posed by the scene's mean gait."""
        cfg = self.config
        metadata = self.kwargs["metadata"]
        cycle = self.gait_cycle(trajectories)
        seeds = self.deform_seeds()
        keypoints, sweeps, lidar_times = self.far_evidence()
        all_times = self.dynamic_actors.unique_timestamps
        _, present = self.dynamic_actors.get_boxes2world(all_times, flatten=False)
        present = present.reshape(len(all_times), -1)
        identity = [1.0, 0.0, 0.0, 0.0]
        moved = []
        for index, trajectory in enumerate(trajectories):
            if index in replaced or trajectory["label"] not in PEDESTRIANS or trajectory["stationary"]:
                continue
            positions = trajectory["poses"][:, :3, 3]
            travel = float((positions[1:] - positions[:-1]).norm(dim=-1).sum())
            route = self.actor_routes[index]
            frame_ids = present[:, index].nonzero().flatten()
            if travel < cfg.gait_min_travel or len(frame_ids) < 2:
                continue
            seconds = float(all_times[frame_ids[-1]] - all_times[frame_ids[0]])
            if travel < cfg.gait_min_speed * seconds:
                route["gait_skipped"] = f"moves {travel / seconds:.2f} m/s, slower than {cfg.gait_min_speed} m/s"
                continue
            if cycle is None:
                route["gait_skipped"] = "no near SMPL walker with a closed leg cycle and no gait library"
                continue
            uuid = trajectory["uuid"]
            times = all_times[frame_ids]
            box_at = self.box_reader(index)
            rotations, centers = (value.detach().float() for value in box_at(times))
            height = float(self.dynamic_actors.actor_sizes[index][2])
            betas = stature_betas(body, height)
            track = {"uuid": uuid, "betas": betas.tolist(), "frames": [
                {"valid": True, "frame_idx": int(f), "timestamp_s": float(t), "root_translation": [0.0, 0.0, 0.0],
                 "root_quat_wxyz": identity, "body_quats_wxyz": [identity] * 23}
                for f, t in zip(frame_ids, times)]}
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(torch.initial_seed() + index)
                person = SMPLTrack(track, body, torch.eye(4), 0.0, feature_dim=cfg.feature_dim,
                                   opacity=cfg.smpl_init_opacity, optimize=True)
            person.anchor_to(box_at)
            fit = FarFit(person, keypoints.get(uuid, []), sweeps.get(uuid, []), metadata["scene_from_world"],
                         float(metadata["scene_time_offset"]), lidar_times, height, cfg.keypoint_min_score)
            decided = choose_phase(person, fit, cycle, times, rotations, centers, height, cfg.gait_phase_steps)
            person.exact_velocity = True
            if cfg.far_ground:
                # Every pose stands on its annotated cuboid floor, which the stance foot starts on.
                person.constraints = PoseConstraints(person, set(person.frame_ids.tolist()),
                                                     floor_planes(person, height), torch.eye(4), prefit=False)
                route.update(ground="cuboid_floor", stance_frames=int(person.constraints.contact.any(dim=-1).sum()))
            route["rigid_gaussians_replaced"] = route["gaussians"]
            route.update(representation="smpl_gait", gaussians=len(person.means), follows_box=True,
                         box_height_m=round(height, 3), shape_beta0=round(float(betas[0]), 3),
                         travel_m=round(travel, 2), gait_source=cycle["source"], gait_donors=cycle["donors"],
                         gait_period_s=round(cycle["period_s"], 3), gait_closing_deg=cycle["closing_deg"],
                         gait_swing_deg=cycle["swing_deg"], **decided)
            dataset = seeds.get(uuid)
            if dataset is not None and len(dataset[0]) >= cfg.deform_seed_min_points:
                points, colours, source = dataset
                route.update(colour_seed_points=len(points), colour_seed_lidar=int((source == 0).sum()),
                             colour_matched=round(paint(person, points, colours, len(times) // 2), 3))
            self.person_index[uuid] = len(self.people)
            if fit.keypoint_observations or fit.lidar_frames:
                self.far_fits[str(len(self.people))] = fit
            self.people.append(person)
            moved.append(index)
        return moved

    def scene_gaussians(self, time):
        gaussians = super().scene_gaussians(time)
        if self.training:
            self.last_actor_time = time.detach()
        if not self.people:
            return gaussians
        articulated = [node.get_gaussians_at(time, self.human_motion) for node in self.people]
        return {name: torch.cat([value, *(person[name] for person in articulated)])
                for name, value in gaussians.items()}

    # SplatAD's step_post_backward runs unchanged: gsplat's MCMC only edits gauss_params (background
    # and rigid actors), never the SMPL Gaussians appended after them.

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
        losses.update(self.far_losses())
        return losses

    def far_losses(self):
        """Keypoint and LiDAR terms of the far walkers, each averaged over the walkers that have it."""
        cfg = self.config
        keypoint, lidar = [], []
        for slot, fit in self.far_fits.items():
            person = self.people[int(slot)]
            if cfg.keypoint_weight:
                keypoint.append(fit.keypoint_loss(person))
            if cfg.lidar_fit_weight:
                lidar.append(fit.lidar_loss(person, cfg.lidar_fit_frames))
        losses = {}
        keypoint, lidar = [v for v in keypoint if v is not None], [v for v in lidar if v is not None]
        if keypoint:
            losses["r3_keypoint"] = torch.stack(keypoint).mean() * cfg.keypoint_weight
        if lidar:
            losses["r3_lidar_fit"] = torch.stack(lidar).mean() * cfg.lidar_fit_weight
        return losses

    def get_param_groups(self):
        groups = super().get_param_groups()
        if self.people:
            for name in GAUSSIAN_PARAMETERS:
                groups[f"smpl_{name}"] = [getattr(person, name) for person in self.people]
            for name, attr in zip(POSE_GROUPS, ("translation", "root", "body"), strict=True):
                groups[name] = [getattr(person, attr) for person in self.people]
        return groups
