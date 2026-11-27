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


def shared_settings(processed, scene, steps, seed, checkpoint_every):
    """Overrides that make OmniRe see the data and budget of the SplatAD runs."""
    return [
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


def prepare(args):
    scene_dir = args.processed / args.scene
    marker = scene_dir / "research-prepared.json"
    smpl = args.pose / args.scene / "humanpose" / "smpl.pkl"
    instances = args.pose / args.scene / "instances" / "instances_info.json"
    expected = {"process_keys": list(PROCESS_KEYS), "smpl_sha256": sha256(smpl)}
    if marker.is_file() and read_json(marker) == expected:
        print(f"OmniRe input for {args.scene} is ready in {scene_dir}")
        return
    from pandaset.annotations import Cuboids

    # The pinned devkit scans for .pkl, while the staged PandaSet stores .pkl.gz (as in omnire-pose.sh).
    Cuboids._data_file_extension = property(lambda self: "pkl.gz")
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


def train(args):
    folder = log_dir(args.output, args.run_id)
    if (folder / "checkpoint_final.pth").is_file():
        print(f"OmniRe already finished: {folder / 'checkpoint_final.pth'}")
        return
    if folder.exists():
        # DriveStudio resumes model weights only and then stops saving, so an interrupted run starts again.
        stale = folder.with_name(f"{folder.name}-interrupted-{time.strftime('%Y%m%d-%H%M%S')}")
        folder.rename(stale)
        print(f"Moved the interrupted OmniRe run to {stale}")
    link_smpl(args.smpl)
    run_module("tools/train.py", [
        "--config_file", "configs/omnire.yaml", "--output_root", args.output,
        "--project", "omnire", "--run_name", args.run_id,
        *shared_settings(args.processed, args.scene, args.steps, args.seed, args.checkpoint_every),
    ], path=True)
    if not (folder / "checkpoint_final.pth").is_file():
        raise RuntimeError(f"OmniRe did not write {folder / 'checkpoint_final.pth'}")


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


def score(args):
    import numpy as np
    import torch
    from PIL import Image
    from pytorch_msssim import SSIM

    from models.video_utils import render_images

    from .evaluate import project
    from .pose_monitor import compare_meshes, select_observations, track_observations
    from .prepare import front_pose_summary
    from .quality import QualityHistory, score_images

    folder = log_dir(args.output, args.run_id)
    cfg, dataset, trainer = load_trainer(folder)
    cameras = list(cfg.data.pixel_source.cameras)
    examples = read_json(args.quality_set)
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    num_cams = dataset.pixel_source.num_cams
    raw_scene = args.raw / args.scene

    def render(sample):
        index = sample["frame"] * num_cams + cameras.index(OMNIRE_CAMERAS.index(sample["camera"]))
        result = render_images(trainer, dataset.full_image_set, vis_indices=[index])
        pred = torch.from_numpy(np.asarray(result["rgbs"][0], dtype=np.float32)).cuda()
        # SplatAD's targets: the original JPEG, back camera without its bottom 260 rows.
        with Image.open(raw_scene / "camera" / f"{sample['camera']}_camera" / f"{sample['frame']:02d}.jpg") as image:
            target = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255).cuda()
        return pred[:sample["height"]], target[:sample["height"]]

    def original(camera, frame):
        with Image.open(raw_scene / "camera" / f"{camera}_camera" / f"{frame:02d}.jpg") as image:
            return torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255).cuda()

    # The same mesh observations as the R2/R3 pose monitor: reviewed ones, then one per SMPL pedestrian.
    manifest = read_json(args.manifest)
    instances_info = json.loads((args.processed / args.scene / "instances" / "instances_info.json").read_text())
    people = smpl_people(dataset, cfg, instances_info) if "SMPLNodes" in cfg.model else []
    slots = {uuid: slot for slot, (_, uuid) in enumerate(people)}
    views = {(name, frame): None for name in OMNIRE_CAMERAS for frame in range(dataset.num_img_timesteps)}
    probe_rows = select_observations(manifest, type("People", (), {"person_index": slots}), views)
    rows = [*probe_rows, *track_observations(manifest, slots, views, {row["uuid"] for row in probe_rows})]
    observations = {f"{row['camera']}/{row['frame']}/{row['uuid']}": {**row, "probe_update": False} for row in rows}
    monitor = {"manifest_sha256": sha256(args.manifest), "observations": observations, "steps": []}
    initial = None

    def capture_meshes(step):
        node = trainer.models["SMPLNodes"]
        if len(people) != node.num_instances:
            raise ValueError(f"Expected {node.num_instances} OmniRe SMPL instances, matched {len(people)} UUIDs")
        results = {}
        for i, (key, row) in enumerate(observations.items()):
            slot = slots[row["uuid"]]
            index = row["frame"] * num_cams + cameras.index(OMNIRE_CAMERAS.index(row["camera"]))
            _, cam_infos = dataset.full_image_set.get_image(index, 1)
            view = torch.linalg.inv(cam_infos["camera_to_world"].float().cuda())
            intrinsics = cam_infos["intrinsics"].float().cuda()
            meshes = []
            for params in (initial, None):
                world, present, faces = posed_meshes(node, row["frame"], params)
                meshes.append((project(world[slot], view, intrinsics), bool(present[slot])))
            rendered = render_images(trainer, dataset.full_image_set, vis_indices=[index])["rgbs"][0]
            prediction = torch.from_numpy(np.asarray(rendered, dtype=np.float32)).cuda()
            values, images = compare_meshes(original(row["camera"], row["frame"]), prediction, row, *meshes, faces)
            written = {}
            for name, image in images.items():
                relative = Path(f"step-{step:06d}") / f"observation-{i + 1}" / f"{name}.png"
                (args.monitor / relative).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(args.monitor / relative)
                written[name] = relative.as_posix()
            results[key] = {"metrics": values, "panels": written, "pose_only_rgb_probe": {"status": "tracked"},
                            "actual_rgb_step": None}
        tracks = {uuid: omnire_drift(node, initial, slot) for uuid, slot in slots.items()}
        monitor["steps"].append({"step": step, "results": results, "tracks": tracks})
        dump_json(args.monitor / "monitor.json", monitor)
        dump_json(args.monitor / "run.json", {
            "scene": args.scene, "variant": "OmniRe", "mode": "pose_monitor", "run_id": args.run_id, "videos": [],
            "monitor": "monitor.json", "pose_summary": front_pose_summary(manifest), "step": step,
            "snapshots": len(monitor["steps"])})

    history = QualityHistory(args.quality_set, args.result, examples["scene_id"], "OmniRe", args.run_id, 0)
    history.steps.clear()
    for step, path in checkpoints(folder, args.steps):
        trainer.resume_from_checkpoint(ckpt_path=str(path), load_only_model=True)
        trainer.set_eval()
        with torch.no_grad():
            history.add_step(step, score_images(examples, render, ssim.cuda()))
            if people:
                if initial is None:
                    device = trainer.models["SMPLNodes"].instances_trans.device
                    initial = initial_pose(dataset.pixel_source, people, device)
                capture_meshes(step)
    if not history.steps:
        raise FileNotFoundError(f"No OmniRe checkpoint in {folder}")


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
    commands.choices["train"].add_argument("--smpl", type=Path, required=True)
    commands.choices["train"].add_argument("--seed", type=int, default=42)
    commands.choices["train"].add_argument("--checkpoint-every", type=int, default=2500)
    commands.choices["score"].add_argument("--raw", type=Path, required=True)
    commands.choices["video"].add_argument("--raw", type=Path, required=True)
    commands.choices["video"].add_argument("--manifest", type=Path, required=True)
    commands.choices["video"].add_argument("--result", type=Path, required=True)
    commands.choices["video"].add_argument("--fps", type=int, default=60)
    commands.choices["video"].add_argument("--crf", type=int, default=16)
    commands.choices["score"].add_argument("--quality-set", type=Path, required=True)
    commands.choices["score"].add_argument("--result", type=Path, required=True)
    commands.choices["score"].add_argument("--manifest", type=Path, required=True)
    commands.choices["score"].add_argument("--monitor", type=Path, required=True,
                                           help="Dashboard folder for the projected OmniRe SMPL meshes")
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
