"""Freeze PandaSet RGB examples and log repeatable PSNR/SSIM during training."""
import argparse
import math
from pathlib import Path

import numpy as np

from .prepare import CAMERAS, PEDESTRIANS, box_records, camera_at, dump_json, projected_box, read_json, sha256


VEHICLES = {
    "Bicycle", "Bus", "Car", "Emergency Vehicle", "Medium-sized Truck", "Motorcycle",
    "Motorized Scooter", "Other Vehicle - Construction Vehicle", "Other Vehicle - Pedicab",
    "Other Vehicle - Uncommon", "Personal Mobility Device", "Pickup Truck", "Semi-truck",
}
SCOPES = ("scene", "vehicle", "person")


def freeze_examples(raw_scene, cameras, frames_per_camera, manifest=None):
    """Sample across each camera sequence independently of model predictions."""
    from PIL import Image

    if manifest is not None and manifest["scene_id"] != raw_scene.name:
        raise ValueError("Pedestrian review and RGB examples must use the same scene")
    observations = (manifest or {}).get("observations", [])
    reviews = {}
    for row in observations:
        if row.get("pose_monitor") and row["camera"] in cameras:
            reviews.setdefault((row["camera"], row["frame"]), {})[row["uuid"]] = row["bbox_xyxy"]
    images = []
    cuboids = {}
    for camera in cameras:
        path = raw_scene / "camera" / f"{camera}_camera"
        timestamps = read_json(path / "timestamps.json")
        frames = set(map(int, np.linspace(0, len(timestamps) - 1, min(frames_per_camera, len(timestamps)), dtype=int)))
        extra_frames = {frame for name, frame in reviews if name == camera}
        for frame in sorted(frames | extra_frames):
            with Image.open(path / f"{frame:02d}.jpg") as image:
                width, height = image.size
            if camera == "back":
                height = min(820, height)
            view = camera_at(raw_scene, camera, frame)
            person_only = frame not in frames
            if not person_only and frame not in cuboids:
                cuboids[frame] = list(box_records(raw_scene, frame, VEHICLES | PEDESTRIANS))
            crops = {
                uuid: {"scope": "person", "uuid": uuid, "label": "Pedestrian", "bbox": bounds}
                for uuid, bounds in reviews.get((camera, frame), {}).items()
            }
            for box in (() if person_only else cuboids[frame]):
                category = "person" if box["label"] in PEDESTRIANS else "vehicle"
                center = np.array([box[f"position.{axis}"] for axis in "xyz"])
                size = np.array([box[f"dimensions.{axis}"] for axis in "xyz"])
                bounds = projected_box(center, size, float(box["yaw"]), view, height, width)
                if bounds and min(bounds[2] - bounds[0], bounds[3] - bounds[1]) >= 8:
                    uuid = str(box["uuid"])
                    crops.setdefault(uuid, {"scope": category, "uuid": uuid, "label": box["label"], "bbox": bounds})
            images.append({"camera": camera, "frame": frame, "width": width, "height": height,
                           "person_only": person_only, "crops": list(crops.values())})
    return {"schema": 2, "scene_id": raw_scene.name, "cameras": list(cameras), "images": images}


def measure_quality(pipeline, examples):
    import torch

    from .evaluate import PERSON_MIN_CROP, camera_indices, camera_sample, crop_bounds, psnr_ssim, rgb_batch

    model = pipeline.model
    manager = pipeline.datamanager
    indices = camera_indices(manager)
    totals = {scope: dict(samples=0, psnr_db=0.0, ssim=0.0) for scope in SCOPES}

    def add(scope, pred, target):
        scores = psnr_ssim(pred, target, model.ssim)
        totals[scope]["samples"] += 1
        for metric, value in scores.items():
            totals[scope][metric] += value

    saved = {key: getattr(model, key) for key in ("info", "last_size") if hasattr(model, key)}
    was_training = pipeline.training
    pipeline.eval()
    try:
        with torch.no_grad():
            for sample in examples["images"]:
                camera, image = camera_sample(manager, indices[sample["camera"], sample["frame"]])
                outputs = model.get_camera_outputs(camera)
                target = model.composite_with_background(model.get_gt_img(image), outputs["background"])
                pred = outputs["rgb"]
                if pred.shape != target.shape or pred.shape[:2] != (sample["height"], sample["width"]):
                    raise ValueError(f"RGB quality image size changed for {sample['camera']}/{sample['frame']}")
                a, b = rgb_batch(pred), rgb_batch(target)
                if not sample.get("person_only"):
                    add("scene", a, b)
                for crop in sample["crops"]:
                    minimum = PERSON_MIN_CROP if examples.get("schema", 1) == 2 and crop["scope"] == "person" else 64
                    x1, y1, x2, y2 = crop_bounds(crop["bbox"], sample["height"], sample["width"], minimum)
                    patch = np.s_[:, :, y1:y2, x1:x2]
                    add(crop["scope"], a[patch].contiguous(), b[patch].contiguous())
    finally:
        pipeline.train(was_training)
        for key, value in saved.items():
            setattr(model, key, value)
    result = {}
    for scope, total in totals.items():
        count = total["samples"]
        scores = {metric: float(total[metric] / count) if count else None for metric in ("psnr_db", "ssim")}
        if any(value is not None and not math.isfinite(value) for value in scores.values()):
            raise ValueError(f"Nonfinite RGB quality score for {scope}")
        result[scope] = {"samples": count, **scores}
    return result


class QualityHistory:
    def __init__(self, sample_path, output, scene, variant, run_id, start_step):
        self.examples = read_json(sample_path)
        if self.examples["schema"] not in (1, 2) or self.examples["scene_id"] != scene or not self.examples["images"]:
            raise ValueError("Invalid frozen RGB quality samples for this scene")
        self.output = output
        self.digest = sha256(sample_path)
        self.record = {"scene": scene, "variant": variant, "run_id": run_id, "mode": "rgb_quality",
                       "quality": "quality.json", "quality_sha256": self.digest, "videos": []}
        self.steps = []
        existing = output / self.record["quality"]
        if existing.is_file():
            report = read_json(existing)
            identity = (scene, variant, run_id, self.digest)
            if tuple(report[key] for key in ("scene", "variant", "run_id", "quality_sha256")) != identity:
                raise ValueError("RGB quality report belongs to another run or sample set")
            self.steps = [row for row in report["steps"] if row["step"] <= start_step]
            if not start_step:
                self.steps.clear()
        self.publish()

    def publish(self):
        identity = {key: self.record[key] for key in ("scene", "variant", "run_id", "quality_sha256")}
        dump_json(self.output / self.record["quality"], {**identity, "steps": self.steps})
        dump_json(self.output / "run.json", {**self.record, "step": self.steps[-1]["step"] if self.steps else None,
                                           "snapshots": len(self.steps)})

    def capture(self, pipeline, step):
        if self.steps and self.steps[-1]["step"] == step:
            return
        summary = measure_quality(pipeline, self.examples)
        self.steps.append({"step": step, "scopes": summary})
        self.publish()
        print(f"RGB quality at {step} optimizer updates: {summary}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+", required=True, choices=CAMERAS)
    parser.add_argument("--frames-per-camera", type=int, default=8)
    parser.add_argument("--manifest", type=Path, help="Include the reviewed pedestrian observations in the fixed crops")
    args = parser.parse_args()
    if args.frames_per_camera < 1:
        parser.error("--frames-per-camera must be positive")
    examples = freeze_examples(args.raw / args.scene, list(dict.fromkeys(args.cameras)), args.frames_per_camera,
                               read_json(args.manifest) if args.manifest else None)
    dump_json(args.output, examples)
    print(f"{args.scene}: {len(examples['images'])} fixed RGB samples")


if __name__ == "__main__":
    main()
