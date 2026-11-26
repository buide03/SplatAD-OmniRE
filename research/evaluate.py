"""Evaluate fixed observations on a checkpoint, with linked image and scan evidence."""
import argparse
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .geometry import matrix_quat, quat_matrix
from .prepare import dump_json, front_pose_summary, read_json, sha256


def camera_indices(manager):
    return {(p.parent.name.removesuffix("_camera"), int(p.stem)): i
            for i, p in enumerate(manager.train_dataset.image_filenames)}


def camera_sample(manager, index, with_image=True):
    image = manager.cached_train[index]["image"].to(manager.device) if with_image else None
    camera = manager.train_dataset.cameras[index:index + 1].to(manager.device)
    camera.metadata = dict(camera.metadata or {}, cam_idx=index)
    if image is not None:
        # PandaSet's back camera uses the calibrated upper 820 rows.
        image = image[: int(camera.height.item()), : int(camera.width.item())]
    return camera, image


def lidar_sample(manager, index, training=False):
    dataset = manager.train_lidar_dataset if training else manager.eval_lidar_dataset
    cached = manager.cached_lidar_train if training else manager.cached_lidar_eval
    lidar = dataset.lidars[index:index + 1].to(manager.device)
    lidar.metadata = dict(lidar.metadata or {}, lidar_idx=index)
    batch = cached[index].copy()
    manager._add_metadata(lidar, batch, len(manager.train_dataset))
    return lidar, batch


def crop_bounds(box, height, width, minimum=64):
    x1, y1, x2, y2 = box
    def interval(start, end, limit):
        length = min(limit, max(minimum, math.ceil(end) - math.floor(start)))
        left = min(max(0, math.floor((start + end - length) / 2)), limit - length)
        return left, left + length
    left, right = interval(x1, x2, width)
    top, bottom = interval(y1, y2, height)
    return left, top, right, bottom


def rgb_scores(pred, gt, lpips, ssim):
    a, b = (t.clamp(0, 1).permute(2, 0, 1)[None].contiguous() for t in (pred, gt))
    diff = a - b
    return {"mae": float(diff.abs().mean()), "psnr_db": float(-10 * diff.square().mean().clamp_min(1e-12).log10()),
            "ssim": float(ssim(a, b)), "lpips": float(lpips(a, b))}


def to_image(image):
    return (image.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


def project(points, view, k):
    local = points @ view[:3, :3].T + view[:3, 3]
    pixel = local @ k.T
    return pixel[:, :2] / pixel[:, 2:].clamp_min(1e-5), local[:, 2] > 0.5


def overlay(image, mesh, visible, faces, color):
    import cv2

    if not visible:
        return image.copy()
    xy, depth_valid = mesh
    valid = depth_valid[faces].all(dim=-1)
    triangles = xy[faces[valid]].clamp(-32000, 32000).cpu().numpy().astype(np.int32)
    layer = image.copy()
    if len(triangles):
        cv2.polylines(layer, triangles, True, color, 1, cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.58, image, 0.42, 0)


def lidar_evidence(model, lidar, batch, observations, scene_from_world):
    raster = batch["raster_pts"].reshape(-1, 5)
    ids = batch["raster_pts_valid_depth_and_did_return"].reshape(-1)
    measured = raster[ids]
    outputs = model.get_lidar_outputs(lidar)
    pred = outputs["depth"].reshape(-1)[ids]
    residual = (pred - measured[:, 2]).abs()
    keep = residual < residual.quantile(model.config.depth_loss_quantile_threshold) if len(ids) else residual.bool()
    az, el = measured[:, :2].deg2rad().unbind(-1)
    unit = torch.stack((el.cos() * az.cos(), el.cos() * az.sin(), el.sin()), dim=-1)
    local = measured[:, 2:3] * unit + measured[:, 3:4] * batch["linear_velocities_local"]
    pose = lidar.lidar_to_worlds[0]
    points = local @ pose[:3, :3].T + pose[:3, 3]
    scene_from_world = scene_from_world.to(points)
    points = (points - scene_from_world[:3, 3]) @ scene_from_world[:3, :3]
    values = {}
    for row in observations:
        uuid = row["uuid"]
        if uuid not in values:
            box = points - points.new_tensor(row["box_center"])
            cos, sin = math.cos(row["box_yaw"]), math.sin(row["box_yaw"])
            rot = points.new_tensor([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
            mask = ((box @ rot).abs() <= points.new_tensor(row["box_size"]) / 2).all(-1)
            hit_count = int(mask.sum())
            values[uuid] = {
                "range_mae_m": float(residual[mask].mean()) if hit_count else None,
                "lidar_hits": hit_count, "lidar_hits_after_quantile": int((mask & keep).sum()),
            }
    gt_range = raster[:, 2].reshape(batch["raster_pts"].shape[:-1])
    pred_range = outputs["depth"].squeeze(-1)
    valid = (gt_range > 0) & (gt_range <= lidar.valid_lidar_distance_threshold)
    return values, (gt_range, pred_range, valid)


def lidar_panel(images, path):
    import cv2
    from PIL import Image

    gt, pred, valid = (array[0].detach().cpu().numpy() for array in images)
    distance_scale = max(30, np.percentile(gt[valid], 95)) if valid.any() else 30
    panels = []
    for label, array, maximum in (
        ("PandaSet range", gt, distance_scale), ("SplatAD range", pred, distance_scale),
        ("Absolute error", np.abs(gt - pred), 3),
    ):
        bars = np.clip(array / maximum, 0, 1) * 255
        color = cv2.applyColorMap(bars.astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
        color[~valid] = 0
        color = cv2.resize(color, (1200, 140), interpolation=cv2.INTER_NEAREST)
        color = np.pad(color, ((25, 0), (0, 0), (0, 0)))
        cv2.putText(color, label, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(color)
    Image.fromarray(np.concatenate(panels)).save(path)


def projected_meshes(person, outputs, camera, gt):
    view = outputs["world_to_camera"]
    k = camera.get_intrinsics_matrices()[0]
    initial, initial_valid = person.mesh_at(outputs["capture_time"], initial=True)
    current, current_valid = person.mesh_at(outputs["capture_time"])
    initial_xy, current_xy = project(initial, view, k), project(current, view, k)
    return (
        overlay(gt, initial_xy, bool(initial_valid), person.faces, (252, 192, 89)),
        overlay(gt, current_xy, bool(current_valid), person.faces, (100, 246, 181)),
        initial_xy, current_xy, bool(initial_valid & current_valid),
    )


def reprojection(labels, meshes):
    values = [[], []]
    if not meshes[-1]:
        return {"keypoint_count": 0}
    for label in labels.values():
        index = int(label["vertex_id"])
        ref = label["xy"]
        if not 0 <= index < len(meshes[2][0]) or not all(bool(p[1][index]) for p in meshes[2:4]):
            continue
        for xy, errors in zip(meshes[2:4], values, strict=True):
            errors.append(float((xy[0][index] - xy[0].new_tensor(ref)).abs().mean()))
    return {"keypoint_count": len(values[0]),
            "reprojection_initial_mae_px": float(np.mean(values[0])) if values[0] else None,
            "reprojection_mae_px": float(np.mean(values[1])) if values[1] else None}


def foot_contacts(model, segments, cameras, manager, offset):
    result = []
    for segment in segments:
        index = model.person_index.get(segment["uuid"])
        if index is None:
            continue
        person = model.people[index]
        vertex = int(segment["vertex_id"])
        if not 0 <= vertex < len(person.vertex_ids):
            raise ValueError(f"Foot contact vertex {vertex} is out of bounds")
        camera = segment["camera"]
        start = cameras[(camera, segment["start_frame"])]
        end = cameras[(camera, segment["end_frame"])]
        cams = manager.train_dataset.cameras
        times = cams.times[[start, end]].to(model.device) + float(offset)
        a, visible_a = person.mesh_at(times[0])
        b, visible_b = person.mesh_at(times[1])
        if bool(visible_a & visible_b):
            result.append({**segment, "foot_slip_cm": float((b[vertex, :2] - a[vertex, :2]).norm() * 100)})
    return result


@torch.no_grad()
def evaluate(pipeline, manifest, output, probes=False, write_panels=True):
    from PIL import Image
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    output.mkdir(parents=True, exist_ok=True)
    manager, model = pipeline.datamanager, pipeline.model
    was_training = pipeline.training
    pipeline.eval()
    try:
        cameras = camera_indices(manager)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(model.device).eval()
        observations = [r for r in manifest["observations"] if r["usable"] and (not probes or r["probe"])]
        selected = set(manifest["selected_uuids"])
        groups, scans = defaultdict(list), defaultdict(list)
        for row in observations:
            groups[row["camera"], row["frame"]].append(row)
            scans[row["scan"]].append(row)
        scan_results = {}
        for scan, items in sorted(scans.items()):
            lidar, batch = lidar_sample(manager, scan)
            values, raster = lidar_evidence(model, lidar, batch, items, model.kwargs["metadata"]["scene_from_world"])
            if write_panels:
                panel = f"lidar/{scan:03d}.jpg"
                (output / panel).parent.mkdir(parents=True, exist_ok=True)
                lidar_panel(raster, output / panel)
            else:
                panel = None
            for uuid, value in values.items():
                scan_results[scan, uuid] = {**value, "lidar_panel": panel}

        scene_metrics, crops = [], []
        for (name, frame), items in sorted(groups.items()):
            camera, image = camera_sample(manager, cameras[name, frame])
            outputs = model.get_camera_outputs(camera)
            pred, gt = outputs["rgb"], model.get_gt_img(image)
            if pred.shape != gt.shape:
                raise ValueError(f"Calibration/image shape mismatch for {name} frame {frame}: {pred.shape} vs {gt.shape}")
            gt_image, pred_image = to_image(gt), to_image(pred)
            scene_metrics.append({"camera": name, "frame": frame, **rgb_scores(pred, gt, lpips, model.ssim)})
            for number, row in enumerate(sorted(items, key=lambda r: r["uuid"])):
                x1, y1, x2, y2 = crop_bounds(row["bbox_xyxy"], *gt.shape[:2])
                roi = np.s_[y1:y2, x1:x2]
                values = rgb_scores(pred[roi], gt[roi], lpips, model.ssim)
                mesh = None
                if row["uuid"] in model.person_index:
                    person = model.people[model.person_index[row["uuid"]]]
                    mesh = projected_meshes(person, outputs, camera, gt_image)
                    values.update(reprojection(row["keypoints"], mesh))
                panels = {}
                if write_panels:
                    panel_dir = Path("panels") / name / f"{frame:03d}-{number:03d}"
                    (output / panel_dir).mkdir(parents=True, exist_ok=True)
                    error = to_image((gt - pred).abs() * 3)
                    images = {"gt": gt_image, "render": pred_image, "error": error}
                    if mesh is not None and mesh[-1]:
                        images.update(initial=mesh[0], current=mesh[1])
                    for label, data in images.items():
                        path = panel_dir / f"{label}.png"
                        Image.fromarray(data[roi]).save(output / path)
                        panels[label] = path.as_posix()
                crops.append({**row, **values, **scan_results[row["scan"], row["uuid"]],
                              "comparison_eligible": row["uuid"] in selected and (
                                  model.config.variant == "R0" or row["uuid"] in model.person_index),
                              "crop_xyxy": [x1, y1, x2, y2], "panels": panels})
        contacts = foot_contacts(model, manifest.get("foot_contacts", []), cameras, manager, -0.01)
        result = {"scene": manifest["scene_id"], "observations": crops, "scene_metrics": scene_metrics,
                  "coverage": manifest["coverage"], "foot_contacts": contacts}
        dump_json(output / "metrics.json", result)
        return result
    finally:
        pipeline.train(was_training)


def load_checkpoint(config_file, checkpoint=None):
    from nerfstudio.utils.eval_utils import eval_setup

    def update(config):
        config.load_dir = checkpoint.parent if checkpoint else config_file.parent / config.relative_model_dir
        config.load_step = int(checkpoint.stem.split("-")[-1]) if checkpoint else None
        return config

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires the CUDA environment used by training")
    return eval_setup(config_file, update_config_callback=update)


def run_record(config, checkpoint, step, manifest_path, manifest, mode):
    base = config.get_base_dir()
    stats_path = base / "run_stats.json"
    stats = read_json(stats_path) if stats_path.exists() else {}
    return {
        "scene": config.pipeline.model.scene_id, "variant": config.pipeline.model.variant,
        "run_id": config.timestamp, "step": step, "mode": mode, "checkpoint": checkpoint.name,
        "manifest_sha256": sha256(manifest_path), "metrics": "metrics.json",
        "pose_summary": front_pose_summary(manifest) if config.pipeline.model.variant != "R0" else None,
        "training": "training.jsonl" if (base / "training.jsonl").exists() else None,
        "run_stats": stats if stats.get("step") == step else {},
        "environment": read_json(base / "environment.json") if (base / "environment.json").exists() else {},
        "videos": [],
    }


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--no-panels", action="store_true")
    parser.add_argument("--zero-smpl-velocity", action="store_true")
    args = parser.parse_args()
    config, pipeline, checkpoint, step = load_checkpoint(args.config.resolve(), args.checkpoint)
    path = args.manifest or config.pipeline.model.manifest_path
    if path is None:
        parser.error("Provide a frozen manifest")
    baseline_hash = config.pipeline.model.manifest_digest
    if baseline_hash and baseline_hash != sha256(path):
        parser.error("The selected manifest differs from the one used in training")
    manifest = read_json(path)
    if manifest["scene_id"] != config.pipeline.model.scene_id:
        parser.error("The manifest belongs to a different scene")
    mode = "zero_smpl_velocity" if args.zero_smpl_velocity else "checkpoint"
    pipeline.model.human_motion = not args.zero_smpl_velocity
    record = run_record(config, checkpoint, step, path, manifest, mode)
    metadata_path = args.output / "run.json"
    if metadata_path.exists():
        existing = read_json(metadata_path)
        keys = ("scene", "variant", "run_id", "step", "mode", "manifest_sha256")
        if any(existing[key] != record[key] for key in keys):
            parser.error("The output directory already belongs to another run")
        record["videos"] = existing["videos"]
        if "front_triptych" in existing:
            record["front_triptych"] = existing["front_triptych"]
        if "interpolation" in existing:
            record["interpolation"] = existing["interpolation"]
    evaluate(pipeline, manifest, args.output, probes=args.probes, write_panels=not args.no_panels)
    if record["training"]:
        source = config.get_base_dir() / record["training"]
        lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
        included = [line for line in lines if json.loads(line)["step"] <= step]
        (args.output / record["training"]).write_text("".join(included), encoding="utf-8")
    dump_json(args.output / "run.json", record)


if __name__ == "__main__":
    main()
