"""Official OmniRe (DriveStudio) baseline on the scene, cameras, LiDAR, poses and RGB samples of SplatAD/R2/R3.

Runs inside ``.deps/omnire-env`` with DriveStudio as the working directory (scripts/omnire-baseline.sh).

* prepare: DriveStudio PandaSet preprocessing, the same 4D-Humans ``smpl.pkl`` as R2/R3, SegFormer-B5 sky masks.
* train:   ``tools/train.py`` with configs/omnire.yaml and pandaset/6cams, unchanged except the shared settings.
* score:   render every saved checkpoint on the frozen RGB samples and score them with research.quality.
"""
import argparse
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path

from .prepare import dump_json, read_json, sha256

# DriveStudio PandaSet camera ids 0-5 (configs/datasets/pandaset/6cams.yaml).
OMNIRE_CAMERAS = ("front", "front_left", "front_right", "left", "right", "back")
PROCESS_KEYS = ("images", "lidar", "calib", "pose", "dynamic_masks", "objects")
REPO = Path(__file__).resolve().parents[1]


def run_module(name, argv, path=False):
    saved = sys.argv
    sys.argv = [name, *map(str, argv)]
    try:
        if path:
            runpy.run_path(name, run_name="__main__")
        else:
            runpy.run_module(name, run_name="__main__")
    finally:
        sys.argv = saved


# The official config loads PandaSet at half resolution. At native resolution its split/duplicate rule grew the
# background to 4.48M Gaussians by step 7,500 of scene 030 and ran a 22 GB L4 out of memory while pruning
# (~350k more every 500 steps). Its schedule stops splitting at 15,000 of 30,000 steps, so a 10,000-step run
# never stops; scaled to 10,000 steps it would stop at 5,000. 7,000 is where the crashed run's resume state
# sits, and leaves ~4.4M Gaussians, near SplatAD's 5M MCMC cap.
STOP_SPLIT_AT = 7000
SPLITTING_CLASSES = ("Background", "RigidNodes", "DeformableNodes")


def shared_settings(processed, scene, steps, seed, checkpoint_every):
    """Overrides that make OmniRe see the data and budget of the SplatAD runs."""
    return [
        *(f"model.{name}.ctrl.stop_split_at={STOP_SPLIT_AT}" for name in SPLITTING_CLASSES),
        "dataset=pandaset/6cams",
        f"data.data_root={processed}",
        f"data.scene_idx={int(scene)}",
        "data.start_timestep=0",
        "data.end_timestep=-1",
        # Native resolution, every frame used for training, as in SplatAD (train_split_fraction=1).
        "data.pixel_source.downscale_when_loading=[1,1,1,1,1,1]",
        "data.pixel_source.test_image_stride=0",
        # SplatAD rasterizes the 360° Pandar64 only.
        "data.lidar_source.only_use_360_lidar=True",
        f"trainer.optim.num_iters={steps}",
        f"seed={seed}",
        f"logging.saveckpt_freq={checkpoint_every}",
    ]


def sky_masks(scene_dir, overwrite=False):
    """SegFormer-B5 Cityscapes 1024x1024, the checkpoint DriveStudio's extract_masks.py uses, via Transformers."""
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    pin = read_json(REPO / "versions.json")["segformer"]
    processor = SegformerImageProcessor.from_pretrained(pin["model"], revision=pin["commit"])
    model = SegformerForSemanticSegmentation.from_pretrained(pin["model"], revision=pin["commit"]).cuda().eval()
    sky = model.config.label2id["sky"]
    output = scene_dir / "sky_masks"
    output.mkdir(exist_ok=True)
    images = sorted((scene_dir / "images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No processed images in {scene_dir / 'images'}")
    for path in images:
        target = output / f"{path.stem}.png"
        if target.is_file() and not overwrite:
            continue
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        inputs = processor(images=rgb, return_tensors="pt", do_resize=False)
        with torch.no_grad():
            logits = model(pixel_values=inputs["pixel_values"].cuda()).logits
            logits = F.interpolate(logits, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False)
        mask = (logits.argmax(1)[0] == sky).cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask).save(target.with_suffix(".partial.png"))
        target.with_suffix(".partial.png").replace(target)
    return len(images)


def read_gzipped_pandaset():
    """The pinned devkit scans cuboids and LiDAR sweeps as *.pkl; the staged PandaSet stores *.pkl.gz.

    Without the LiDAR override the sweep list is empty and DriveStudio's preprocess fails on frame 0
    with "list index out of range". pandas infers gzip from the suffix, and the preprocess workers
    are forked, so they inherit both overrides.
    """
    from pandaset.annotations import Cuboids
    from pandaset.sensors import Lidar

    for sensor in (Cuboids, Lidar):
        sensor._data_file_extension = property(lambda self: "pkl.gz")


def prepare(args):
    scene_dir = args.processed / args.scene
    marker = scene_dir / "research-prepared.json"
    smpl = args.pose / args.scene / "humanpose" / "smpl.pkl"
    instances = args.pose / args.scene / "instances" / "instances_info.json"
    expected = {"process_keys": list(PROCESS_KEYS), "smpl_sha256": sha256(smpl)}
    if marker.is_file() and read_json(marker) == expected:
        print(f"OmniRe input for {args.scene} is ready in {scene_dir}")
        return
    read_gzipped_pandaset()
    run_module("datasets.preprocess", [
        "--dataset", "pandaset", "--data_root", args.raw, "--target_dir", args.processed,
        "--scene_ids", args.scene, "--workers", "2", "--process_keys", *PROCESS_KEYS,
    ])
    # The SMPL tracks are keyed by the instance ids of the pose run. Both runs must agree.
    if json.loads((scene_dir / "instances" / "instances_info.json").read_text()) != json.loads(instances.read_text()):
        raise ValueError("DriveStudio instance ids differ from the 4D-Humans pose run. Re-run the pose stage.")
    (scene_dir / "humanpose").mkdir(exist_ok=True)
    shutil.copyfile(smpl, scene_dir / "humanpose" / "smpl.pkl")
    count = sky_masks(scene_dir)
    dump_json(marker, expected)
    print(f"OmniRe input for {args.scene}: {count} images with sky masks, poses from {smpl}")


def link_smpl(smpl):
    target = Path("smpl_models") / "SMPL_NEUTRAL.pkl"
    target.parent.mkdir(exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(Path(smpl).resolve())


def log_dir(output, run_id):
    return output / "omnire" / run_id


def checkpoints(folder, steps):
    found = []
    for path in folder.glob("checkpoint_*.pth"):
        label = path.stem.removeprefix("checkpoint_")
        found.append((steps if label == "final" else int(label), path))
    return sorted(dict(found).items())


POSE_PARAMETERS = {"translation": "instances_trans", "root": "instances_quats", "body": "smpl_qauts"}


def smpl_people(dataset, cfg, instances_info):
    """UUIDs of the SMPLNodes instances, in the order of DriveStudio's get_init_smpl_objects."""
    init = cfg.model.SMPLNodes.get("init", {})
    only_moving, threshold = init.get("only_moving", False), init.get("traj_length_thres", 0.5)
    source = dataset.pixel_source
    people = []
    for index in range(dataset.instance_num):
        true_id = source.instances_true_id[index].item()
        human = source.smpl_human_all.get(true_id)
        if human is None or human["frame_valid"].sum() == 0:
            continue
        if only_moving and threshold > 0:
            trans = human["smpl_trans"][human["frame_valid"]]
            if (trans[1:] - trans[:-1]).norm(dim=-1).sum() < threshold:
                continue
        people.append((true_id, instances_info[str(true_id)]["id"]))
    return people


def initial_pose(source, people, device):
    """SMPLNodes.create_from_pcd starts from the 4D-Humans quaternions and cuboid-aligned translations."""
    import torch

    humans = [source.smpl_human_all[true_id] for true_id, _ in people]
    return {
        "instances_quats": torch.stack([h["smpl_quats"][:, 0] for h in humans], 1)[:, :, None].to(device),
        "smpl_qauts": torch.stack([h["smpl_quats"][:, 1:] for h in humans], 1).to(device),
        "instances_trans": torch.stack([h["smpl_trans"] for h in humans], 1).to(device),
    }


def posed_meshes(node, frame, params=None):
    """World-space SMPL vertices of every instance at ``frame``, optionally with other pose parameters."""
    import torch

    saved = {}
    try:
        for name, value in (params or {}).items():
            saved[name] = getattr(node, name).data.clone()
            getattr(node, name).data.copy_(value)
        node.cur_frame = torch.tensor(frame, device=node.instances_fv.device)
        node.in_test_set = False
        vertices, faces = node.template.get_init_vf()
        vertices = vertices.to(node.instances_fv.device).reshape(-1, 3)
        world = node.transform_means(vertices).reshape(node.num_instances, -1, 3)
        return world, node.instances_fv[frame], faces.to(world.device).long()
    finally:
        for name, value in saved.items():
            getattr(node, name).data.copy_(value)


def omnire_drift(node, initial, slot):
    from .pose_monitor import pose_drift

    valid = node.instances_fv[:, slot]
    current = {name: getattr(node, attr)[:, slot][valid] for name, attr in POSE_PARAMETERS.items()}
    reference = {name: initial[attr][:, slot][valid] for name, attr in POSE_PARAMETERS.items()}
    current["root"], reference["root"] = current["root"].squeeze(-2), reference["root"].squeeze(-2)
    return pose_drift(current, reference)


def load_trainer(folder):
    import torch
    from omegaconf import OmegaConf

    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    cfg = OmegaConf.load(folder / "config.yaml")
    if int(cfg.data.start_timestep) != 0:
        raise ValueError("Frame indices assume OmniRe starts at timestep 0")
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model,
        num_train_images=len(dataset.train_image_set), num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda"),
    )
    return cfg, dataset, trainer


def camera_path(c2w, times, samples):
    """The replay.py camera path: PCHIP positions and a rotation spline through the recorded poses."""
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    from scipy.spatial.transform import Rotation, RotationSpline

    recorded = c2w.cpu().double().numpy()
    samples = np.clip(samples, times[0], times[-1])
    position = PchipInterpolator(times, recorded[:, :3, 3], axis=0)(samples)
    rotation = RotationSpline(times, Rotation.from_matrix(recorded[:, :3, :3]))(samples).as_matrix()
    poses = np.tile(np.eye(4), (len(samples), 1, 1))
    poses[:, :3, :3], poses[:, :3, 3] = rotation, position
    return poses


GRADIENT_TAGS = {
    # Same names as the R2/R3 training log, so the dashboard plots all models on one chart.
    "smpl_translation": ("SMPLNodes#ins_translation",),
    "smpl_root_rotation": ("SMPLNodes#ins_rotation",),
    "smpl_body_rotation": ("SMPLNodes#smpl_rotation",),
    "smpl_gaussians": tuple(f"SMPLNodes#{name}" for name in ("xyz", "sh_dc", "sh_rest", "opacity", "scaling", "rotation")),
    "deformable_network": ("DeformableNodes#deform_network",),
    "deformable_embedding": ("DeformableNodes#embedding",),
    "deformable_gaussians": tuple(f"DeformableNodes#{name}" for name in
                                  ("xyz", "sh_dc", "sh_rest", "opacity", "scaling", "rotation")),
    "trajectory": ("RigidNodes#ins_translation", "RigidNodes#ins_rotation"),
    "background_means": ("Background#xyz",),
}


def gradient_row(trainer, step, loss_dict):
    from .pose_monitor import finite_number, gradient_stats

    groups = {group["name"]: group["params"] for group in trainer.optimizer.param_groups}
    return {
        "step": step, "sensor": "rgb",
        "scene_loss": finite_number(float(sum(value.detach() for value in loss_dict.values()))),
        "loss_terms": {name: finite_number(float(value.detach())) for name, value in loss_dict.items()},
        "gradients": {tag: gradient_stats([p for name in names for p in groups.get(name, [])])
                      for tag, names in GRADIENT_TAGS.items()},
    }


def truncate_jsonl(path, start_step):
    if not path.is_file():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        try:
            if json.loads(line)["step"] < start_step:
                kept.append(line)
        except (json.JSONDecodeError, KeyError):
            continue
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(kept), encoding="utf-8")
    temporary.replace(path)


class OmniReMonitor:
    """RGB quality and projected SMPL meshes of a DriveStudio trainer, in the R2/R3 dashboard format."""

    def __init__(self, args, cfg, dataset, trainer, start_step=0):
        from pytorch_msssim import SSIM

        from .pose_monitor import select_observations, track_observations
        from .quality import QualityHistory

        self.args, self.cfg, self.dataset, self.trainer = args, cfg, dataset, trainer
        self.cameras = list(cfg.data.pixel_source.cameras)
        self.num_cams = dataset.pixel_source.num_cams
        self.raw_scene = args.raw / args.scene
        self.examples = read_json(args.quality_set)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).cuda()
        self.history = QualityHistory(args.quality_set, args.result, self.examples["scene_id"], "OmniRe",
                                      args.run_id, start_step)
        self.manifest = read_json(args.manifest)
        instances = json.loads((args.processed / args.scene / "instances" / "instances_info.json").read_text())
        self.people = smpl_people(dataset, cfg, instances) if "SMPLNodes" in cfg.model else []
        self.slots = {uuid: slot for slot, (_, uuid) in enumerate(self.people)}
        views = {(name, frame): None for name in OMNIRE_CAMERAS for frame in range(dataset.num_img_timesteps)}
        probe_rows = select_observations(self.manifest, type("People", (), {"person_index": self.slots}), views)
        rows = [*probe_rows, *track_observations(self.manifest, self.slots, views, {r["uuid"] for r in probe_rows})]
        self.observations = {f"{r['camera']}/{r['frame']}/{r['uuid']}": {**r, "probe_update": False} for r in rows}
        self.monitor = {"manifest_sha256": sha256(args.manifest), "observations": self.observations, "steps": []}
        previous = args.monitor / "monitor.json"
        if start_step and previous.is_file():
            stored = read_json(previous)
            if stored.get("observations") == self.observations:
                self.monitor["steps"] = [entry for entry in stored["steps"] if entry["step"] <= start_step]
        self.initial = None

    def index(self, camera, frame):
        return frame * self.num_cams + self.cameras.index(OMNIRE_CAMERAS.index(camera))

    def original(self, camera, frame):
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(self.raw_scene / "camera" / f"{camera}_camera" / f"{frame:02d}.jpg") as image:
            return torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255).cuda()

    def render(self, index):
        import numpy as np
        import torch

        from models.video_utils import render_images

        rendered = render_images(self.trainer, self.dataset.full_image_set, vis_indices=[index])["rgbs"][0]
        return torch.from_numpy(np.asarray(rendered, dtype=np.float32)).cuda()

    def capture(self, step):
        import torch

        from .quality import score_images

        if self.history.steps and self.history.steps[-1]["step"] >= step:
            return
        self.trainer.set_eval()
        with torch.no_grad():
            def sample_pair(sample):
                # SplatAD's targets: the original JPEG, back camera without its bottom 260 rows.
                pred = self.render(self.index(sample["camera"], sample["frame"]))
                return pred[:sample["height"]], self.original(sample["camera"], sample["frame"])[:sample["height"]]

            self.history.add_step(step, score_images(self.examples, sample_pair, self.ssim))
            if self.people:
                self.capture_meshes(step)

    def capture_meshes(self, step):
        import torch
        from PIL import Image

        from .evaluate import project
        from .pose_monitor import compare_meshes
        from .prepare import front_pose_summary

        node = self.trainer.models["SMPLNodes"]
        if len(self.people) != node.num_instances:
            raise ValueError(f"Expected {node.num_instances} OmniRe SMPL instances, matched {len(self.people)} UUIDs")
        if self.initial is None:
            self.initial = initial_pose(self.dataset.pixel_source, self.people, node.instances_trans.device)
        results = {}
        for i, (key, row) in enumerate(self.observations.items()):
            slot = self.slots[row["uuid"]]
            index = self.index(row["camera"], row["frame"])
            _, cam_infos = self.dataset.full_image_set.get_image(index, 1)
            view = torch.linalg.inv(cam_infos["camera_to_world"].float().cuda())
            intrinsics = cam_infos["intrinsics"].float().cuda()
            meshes = []
            for params in (self.initial, None):
                world, present, faces = posed_meshes(node, row["frame"], params)
                meshes.append((project(world[slot], view, intrinsics), bool(present[slot])))
            values, images = compare_meshes(self.original(row["camera"], row["frame"]), self.render(index),
                                            row, *meshes, faces)
            written = {}
            for name, image in images.items():
                relative = Path(f"step-{step:06d}") / f"observation-{i + 1}" / f"{name}.png"
                (self.args.monitor / relative).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(self.args.monitor / relative)
                written[name] = relative.as_posix()
            results[key] = {"metrics": values, "panels": written, "pose_only_rgb_probe": {"status": "tracked"},
                            "actual_rgb_step": None}
        tracks = {uuid: omnire_drift(node, self.initial, slot) for uuid, slot in self.slots.items()}
        self.monitor["steps"] = [entry for entry in self.monitor["steps"] if entry["step"] < step]
        self.monitor["steps"].append({"step": step, "results": results, "tracks": tracks})
        dump_json(self.args.monitor / "monitor.json", self.monitor)
        dump_json(self.args.monitor / "run.json", {
            "scene": self.args.scene, "variant": "OmniRe", "mode": "pose_monitor", "run_id": self.args.run_id,
            "videos": [], "monitor": "monitor.json", "pose_summary": front_pose_summary(self.manifest),
            "step": step, "snapshots": len(self.monitor["steps"])})


def save_resume_state(path, trainer, step):
    """Model, Adam moments, step and RNG streams, written atomically next to the official checkpoints."""
    import random

    import numpy as np
    import torch

    state = {
        "step": step, "model": trainer.state_dict(only_model=True), "optimizer": trainer.optimizer.state_dict(),
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(), "python": random.getstate()},
    }
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


def train(args):
    """DriveStudio's tools/train.py loop, step for step, plus resume state, live monitoring and logs."""
    import random
    from argparse import Namespace

    import numpy as np
    import torch

    folder = log_dir(args.output, args.run_id)
    if (folder / "checkpoint_final.pth").is_file():
        print(f"OmniRe already finished: {folder / 'checkpoint_final.pth'}")
        return
    state_path = folder / "resume.pth"
    resume = None
    if state_path.is_file():
        try:
            resume = torch.load(state_path, map_location="cpu", weights_only=False)
        except Exception as error:  # A truncated write leaves no usable state.
            print(f"Ignoring unreadable OmniRe resume state: {error}")
    if folder.exists() and resume is None:
        stale = folder.with_name(f"{folder.name}-interrupted-{time.strftime('%Y%m%d-%H%M%S')}")
        folder.rename(stale)
        print(f"No resume state. Moved the earlier OmniRe attempt to {stale}")
    link_smpl(args.smpl)

    from tools import train as upstream
    from datasets.driving_dataset import DrivingDataset
    from models.video_utils import render_images, save_videos
    from utils.logging import MetricLogger
    from utils.misc import import_str

    namespace = Namespace(
        config_file="configs/omnire.yaml", output_root=str(args.output), project="omnire", run_name=args.run_id,
        enable_wandb=False, entity="", resume_from=None, render_video_postfix=None, save_catted_videos=False,
        enable_viewer=False, viewer_port=8080,
        opts=shared_settings(args.processed, args.scene, args.steps, args.seed, args.checkpoint_every),
    )
    cfg = upstream.setup(namespace)
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model,
        num_train_images=len(dataset.train_image_set), num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda"),
    )
    if resume is not None:
        trainer.load_state_dict(resume["model"], load_only_model=True, strict=True)
        smpl = trainer.models.get("SMPLNodes")
        if smpl is not None:
            # DriveStudio does not store the k-NN table that knn_reg reads.
            smpl.update_knn(smpl._means)
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
    render_keys = ["gt_rgbs", "rgbs", "Background_rgbs", "Dynamic_rgbs", "RigidNodes_rgbs",
                   "DeformableNodes_rgbs", "SMPLNodes_rgbs"]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    trainer.initialize_optimizer()
    start = 0
    if resume is not None:
        trainer.optimizer.load_state_dict(resume["optimizer"])
        rng = resume["rng"]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        start = resume["step"] + 1
        print(f"Resuming OmniRe at step {start}", flush=True)
    log = args.result / "training.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    truncate_jsonl(log, start)
    monitor = OmniReMonitor(args, cfg, dataset, trainer, start_step=start)
    if start == 0:
        monitor.capture(0)

    metric_logger = MetricLogger(delimiter="  ", output_file=os.path.join(cfg.log_dir, "metrics.json"))
    step = start
    for step in metric_logger.log_every(np.arange(start, trainer.num_iters + 1), cfg.logging.print_freq):
        step = int(step)
        if step % cfg.logging.vis_freq == 0 and cfg.logging.vis_freq > 0:
            vis_timestep = np.linspace(0, dataset.num_img_timesteps, trainer.num_iters // cfg.logging.vis_freq + 1,
                                       endpoint=False, dtype=int)[step // cfg.logging.vis_freq]
            with torch.no_grad():
                results = render_images(trainer=trainer, dataset=dataset.full_image_set, compute_metrics=True,
                                        compute_error_map=cfg.render.vis_error,
                                        vis_indices=[vis_timestep * dataset.pixel_source.num_cams + i
                                                     for i in range(dataset.pixel_source.num_cams)])
            save_videos(results, save_pth=os.path.join(cfg.log_dir, "images", f"step_{step}.png"),
                        layout=dataset.layout, num_timestamps=1, keys=render_keys,
                        save_seperate_video=cfg.logging.save_seperate_video,
                        num_cams=dataset.pixel_source.num_cams, fps=cfg.render.fps, verbose=False)
            del results
            torch.cuda.empty_cache()

        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad()
        image_infos, cam_infos = dataset.train_image_set.next(trainer._get_downscale_factor())
        image_infos = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in image_infos.items()}
        cam_infos = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in cam_infos.items()}
        outputs = trainer(image_infos, cam_infos)
        trainer.update_visibility_filter()
        loss_dict = trainer.compute_losses(outputs=outputs, image_infos=image_infos, cam_infos=cam_infos)
        for name, value in loss_dict.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                raise ValueError(f"Non-finite loss {name} at step {step}")
        trainer.backward(loss_dict)
        if step % args.log_every == 0:
            # Before postprocessing, which may split or prune Gaussians and their gradients.
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(gradient_row(trainer, step, loss_dict), allow_nan=False) + "\n")
        trainer.postprocess_per_train_step(step=step)

        with torch.no_grad():
            metric_dict = trainer.compute_metrics(outputs=outputs, image_infos=image_infos)
        metric_logger.update(**{"train_metrics/" + k: v.item() for k, v in metric_dict.items()})
        metric_logger.update(**{"train_stats/gaussian_num_" + k: v for k, v in trainer.get_gaussian_count().items()})
        metric_logger.update(**{"losses/" + k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{"train_stats/lr_" + group["name"]: group["lr"]
                                for group in trainer.optimizer.param_groups})

        if step > 0 and (step % cfg.logging.saveckpt_freq == 0 or step == trainer.num_iters):
            trainer.save_checkpoint(log_dir=cfg.log_dir, save_only_model=True, is_final=step == trainer.num_iters)
        if trainer.optim_general.cache_buffer_freq > 0 and step > 0 and step % trainer.optim_general.cache_buffer_freq == 0:
            trainer.set_eval()
            with torch.no_grad():
                dataset.pixel_source.update_downscale_factor(1 / dataset.pixel_source.buffer_downscale)
                results = render_images(trainer=trainer, dataset=dataset.full_image_set)
                dataset.pixel_source.reset_downscale_factor()
                dataset.pixel_source.update_image_error_maps(results)
            del results
            torch.cuda.empty_cache()
        completed = min(step + 1, args.steps)
        if completed % args.monitor_every == 0 or step == trainer.num_iters:
            monitor.capture(completed)
        if (step + 1) % args.resume_every == 0 and step < trainer.num_iters:
            save_resume_state(state_path, trainer, step)

    upstream.do_evaluation(step=step, cfg=cfg, trainer=trainer, dataset=dataset, render_keys=render_keys,
                           args=namespace)
    if not (folder / "checkpoint_final.pth").is_file():
        raise RuntimeError(f"OmniRe did not write {folder / 'checkpoint_final.pth'}")
    state_path.unlink(missing_ok=True)


def score(args):
    """Offline fallback: score every saved checkpoint with the same monitor used during training."""
    import torch

    folder = log_dir(args.output, args.run_id)
    cfg, dataset, trainer = load_trainer(folder)
    monitor = OmniReMonitor(args, cfg, dataset, trainer)
    monitor.history.steps.clear()
    found = checkpoints(folder, args.steps)
    if not found:
        raise FileNotFoundError(f"No OmniRe checkpoint in {folder}")
    for step, path in found:
        trainer.resume_from_checkpoint(ckpt_path=str(path), load_only_model=True)
        with torch.no_grad():
            monitor.capture(step)


def video(args):
    """Front-camera MP4s from the final checkpoint on the 60 FPS clock of the SplatAD replays.

    Cameras follow the interpolated recorded path. OmniRe models time as frame indices
    (SceneGraph picks the nearest normalized timestamp), so the scene state advances at 10 Hz.
    """
    import torch

    from datasets.base.pixel_source import get_rays

    from .evaluate import to_image
    from .prepare import front_pose_summary
    from .sync_video import FRONT_CAMERAS, Timeline, combine_front, ffmpeg_executable, write_video

    folder = log_dir(args.output, args.run_id)
    checkpoint = folder / "checkpoint_final.pth"
    record_path = args.result / "run.json"
    if record_path.is_file():
        record = read_json(record_path)
        triptych = record.get("front_triptych")
        if (record.get("step") == args.steps and record.get("fps") == args.fps and triptych
                and (args.result / triptych["file"]).is_file()):
            print(f"OmniRe replay already exported: {args.result / triptych['file']}")
            return
    cfg, dataset, trainer = load_trainer(folder)
    trainer.resume_from_checkpoint(ckpt_path=str(checkpoint), load_only_model=True)
    trainer.set_eval()
    cameras = list(cfg.data.pixel_source.cameras)
    raw_scene = args.raw / args.scene
    tracks = {name: read_json(raw_scene / "camera" / f"{name}_camera" / "timestamps.json") for name in FRONT_CAMERAS}
    count = dataset.num_img_timesteps
    tracks = {name: [float(t) for t in times[:count]] for name, times in tracks.items()}
    clock = Timeline.from_tracks(list(tracks.values()), args.fps)
    ffmpeg = ffmpeg_executable()
    args.result.mkdir(parents=True, exist_ok=True)
    videos = {}
    for name in FRONT_CAMERAS:
        cam_id = OMNIRE_CAMERAS.index(name)
        if cam_id not in cameras:
            raise ValueError(f"OmniRe was trained without the {name} camera")
        camera = dataset.pixel_source.camera_data[cam_id]
        times = tracks[name]
        samples = list(clock.times())
        poses = torch.from_numpy(camera_path(camera.cam_to_worlds[:count], times, samples)).float().cuda()
        intrinsics = camera.intrinsics[0].float().cuda()
        # Loaded image size, which the intrinsics are scaled to (native size with downscale_when_loading=1).
        height, width = (camera.images.shape[1:3] if getattr(camera, "images", None) is not None
                         else (camera.HEIGHT, camera.WIDTH))
        height, width = int(height), int(width)
        x, y = torch.meshgrid(torch.arange(width, device="cuda"), torch.arange(height, device="cuda"), indexing="xy")
        pixel_coords = torch.stack([y.float() / height, x.float() / width], dim=-1)
        nearest = [min(range(count), key=lambda frame: abs(times[frame] - t)) for t in samples]

        def images():
            with torch.no_grad():
                for pose, frame in zip(poses, nearest, strict=True):
                    origins, viewdirs, direction_norm = get_rays(x.flatten(), y.flatten(), pose, intrinsics)
                    full = lambda value, dtype: torch.full((height, width), value, dtype=dtype, device="cuda")
                    image_infos = {
                        "origins": origins.reshape(height, width, 3),
                        "viewdirs": viewdirs.reshape(height, width, 3),
                        "direction_norm": direction_norm.reshape(height, width, 1),
                        "pixel_coords": pixel_coords,
                        "img_idx": full(int(camera.unique_img_idx[frame]), torch.long),
                        "frame_idx": full(frame, torch.long),
                        "normed_time": full(float(camera.normalized_time[frame]), torch.float32),
                    }
                    cam_infos = {
                        "cam_id": full(camera.cam_id, torch.long), "cam_name": camera.cam_name,
                        "camera_to_world": pose, "intrinsics": intrinsics,
                        "height": torch.tensor(height, device="cuda"), "width": torch.tensor(width, device="cuda"),
                    }
                    yield to_image(trainer(image_infos, cam_infos, novel_view=True)["rgb"])

        path = args.result / f"{name}.mp4"
        frames = write_video(ffmpeg, path, images(), width, height, args.fps, args.crf)
        if frames != clock.frames:
            raise RuntimeError(f"Incomplete OmniRe replay for {name}")
        frames_of = list(range(count))
        videos[name] = {**clock.video_metadata(path, width + width % 2, height + height % 2), "camera": name,
                        "scale": 1, "crf": args.crf, "frame_times": clock.frame_times(frames_of, times)}
        print(f"OmniRe {name}: {frames} frames at {args.fps} FPS", flush=True)
    manifest = read_json(args.manifest)
    record = {
        "scene": args.scene, "variant": "OmniRe", "run_id": args.run_id, "step": args.steps, "mode": "checkpoint",
        "checkpoint": checkpoint.name, "manifest_sha256": sha256(args.manifest), "metrics": None,
        "pose_summary": front_pose_summary(manifest), "training": None, "run_stats": {}, "environment": {},
        "interpolation": "smooth", "fps": args.fps, "videos": list(videos.values()),
        "front_triptych": combine_front(ffmpeg, args.result, videos, args.crf),
    }
    dump_json(record_path, record)
    print(f"OmniRe three synchronized views: {args.result / record['front_triptych']['file']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scene", required=True)
    common.add_argument("--processed", type=Path, required=True)
    step = commands.add_parser("prepare", parents=[common])
    step.add_argument("--raw", type=Path, required=True)
    step.add_argument("--pose", type=Path, required=True, help="Root with <scene>/humanpose/smpl.pkl from the pose stage")
    for name in ("train", "score", "video"):
        step = commands.add_parser(name, parents=[common])
        step.add_argument("--output", type=Path, required=True)
        step.add_argument("--run-id", required=True)
        step.add_argument("--steps", type=int, required=True)
    for name in ("train", "score"):
        step = commands.choices[name]
        step.add_argument("--raw", type=Path, required=True)
        step.add_argument("--quality-set", type=Path, required=True)
        step.add_argument("--result", type=Path, required=True, help="Dashboard folder for quality and training logs")
        step.add_argument("--manifest", type=Path, required=True)
        step.add_argument("--monitor", type=Path, required=True,
                          help="Dashboard folder for the projected OmniRe SMPL meshes")
    commands.choices["train"].add_argument("--smpl", type=Path, required=True)
    commands.choices["train"].add_argument("--seed", type=int, default=42)
    commands.choices["train"].add_argument("--checkpoint-every", type=int, default=2500)
    commands.choices["train"].add_argument("--log-every", type=int, default=25)
    commands.choices["train"].add_argument("--monitor-every", type=int, default=500)
    commands.choices["train"].add_argument("--resume-every", type=int, default=1000)
    commands.choices["video"].add_argument("--raw", type=Path, required=True)
    commands.choices["video"].add_argument("--manifest", type=Path, required=True)
    commands.choices["video"].add_argument("--result", type=Path, required=True)
    commands.choices["video"].add_argument("--fps", type=int, default=60)
    commands.choices["video"].add_argument("--crf", type=int, default=16)
    args = parser.parse_args()
    if not args.scene.isdigit():
        parser.error("PandaSet scene ids are numeric")
    args.scene = f"{int(args.scene):03d}"
    for name in ("processed", "output", "raw", "pose", "smpl", "quality_set", "result", "manifest", "monitor"):
        if getattr(args, name, None) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if Path.cwd().resolve() != (REPO / ".deps" / "omnire").resolve():
        parser.error("Run through scripts/omnire-baseline.sh so DriveStudio is the working directory")
    os.environ.setdefault("WANDB_MODE", "disabled")
    {"prepare": prepare, "train": train, "score": score, "video": video}[args.command](args)


if __name__ == "__main__":
    main()
