"""2D body keypoints of pedestrians for R3's far walkers: ViTPose, top-down on the manifest boxes.

4D-Humans cannot fit a 3D body to a person 30-90 px tall, but a 2D keypoint detector still finds hips,
knees, ankles and wrists there: ViTPose crops each box, resizes it to its 256 x 192 input and returns
17 COCO keypoints with a confidence. Each observation also stores the PandaSet camera (rotation,
centre, intrinsics, timestamp), so research.far_fit can project SMPL joints without the data manager.

Run in the train environment after installing the pinned ``transformers`` (the notebook does both).
"""
import argparse
from pathlib import Path

import numpy as np

from .prepare import camera_at, dump_json, read_json, sha256

SCHEMA_VERSION = 1
MODEL = "usyd-community/vitpose-base-simple"
TRANSFORMERS = "4.53.2"
COCO = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle")
MARGIN = 0.15


def expanded_box(bbox, width, height, margin=MARGIN):
    """A manifest box (x1, y1, x2, y2) grown by ``margin`` per side, clipped, as COCO (x, y, w, h)."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0.0, x1 - dx), max(0.0, y1 - dy)
    x2, y2 = min(float(width), x2 + dx), min(float(height), y2 + dy)
    return [x1, y1, x2 - x1, y2 - y1]


def observations_by_image(manifest):
    """Usable rows (the manifest only holds pedestrians) grouped by (camera, frame)."""
    groups = {}
    for row in manifest["observations"]:
        if row["usable"]:
            groups.setdefault((row["camera"], row["frame"]), []).append(row)
    return groups


def detect(raw, scene_id, manifest, estimator):
    """Keypoints of every usable observation. ``estimator(image, boxes_xywh)`` -> [(xy [17, 2], score [17])]."""
    from PIL import Image

    scene = raw / scene_id
    rows = []
    for (camera, frame), items in sorted(observations_by_image(manifest).items()):
        with Image.open(scene / "camera" / f"{camera}_camera" / f"{frame:02d}.jpg") as image:
            image = image.convert("RGB")
        boxes = [expanded_box(row["bbox_xyxy"], *image.size) for row in items]
        rotation, origin, k, timestamp = camera_at(scene, camera, frame)
        lens = {"fx": float(k["fx"]), "fy": float(k["fy"]), "cx": float(k["cx"]), "cy": float(k["cy"])}
        for row, box, (xy, score) in zip(items, boxes, estimator(image, boxes), strict=True):
            rows.append({
                "uuid": row["uuid"], "camera": camera, "frame": frame, "timestamp_s": float(timestamp),
                "bbox_xyxy": [float(v) for v in row["bbox_xyxy"]], "crop_xywh": [round(v, 2) for v in box],
                "keypoints_xy": np.asarray(xy, dtype=float).round(2).tolist(),
                "scores": np.asarray(score, dtype=float).round(4).tolist(),
                "camera_rotation": np.asarray(rotation, dtype=float).tolist(),
                "camera_origin": np.asarray(origin, dtype=float).tolist(), "intrinsics": lens,
            })
    return rows


def vitpose(device="cuda"):
    """The pinned ViTPose estimator from Hugging Face transformers."""
    import torch
    import transformers
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    if transformers.__version__ != TRANSFORMERS:
        raise RuntimeError(f"Install transformers=={TRANSFORMERS} in this environment (found {transformers.__version__})")

    processor = AutoProcessor.from_pretrained(MODEL)
    model = VitPoseForPoseEstimation.from_pretrained(MODEL).to(device).eval()

    def estimate(image, boxes):
        inputs = processor(image, boxes=[boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_pose_estimation(outputs, boxes=[boxes], threshold=0.0)[0]
        return [scatter(r) for r in results]

    revision = getattr(model.config, "_commit_hash", None)
    return estimate, revision


def scatter(result):
    """(xy [17, 2], score [17]) from one post-processed box, whose keypoints may come with labels."""
    as_array = lambda value: np.asarray(value.cpu() if hasattr(value, "cpu") else value)
    xy, score = np.zeros((len(COCO), 2)), np.zeros(len(COCO))
    labels = result.get("labels")
    labels = np.arange(len(COCO)) if labels is None else as_array(labels).astype(int)
    xy[labels] = as_array(result["keypoints"])
    score[labels] = as_array(result["scores"])
    return xy, score


def load_keypoints(path, scene_id, manifest_digest=None):
    document = read_json(path)
    if document["schema_version"] != SCHEMA_VERSION or document["scene_id"] != str(scene_id):
        raise ValueError("The keypoint file was built for another scene or format. Rebuild it.")
    if manifest_digest and document["manifest_sha256"] != manifest_digest:
        raise ValueError("The keypoint file was built from another manifest. Rebuild it.")
    return document["observations"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    if manifest["scene_id"] != args.scene:
        parser.error("The manifest belongs to another scene")
    estimator, revision = vitpose()
    rows = detect(args.raw, args.scene, manifest, estimator)
    dump_json(args.output, {"schema_version": SCHEMA_VERSION, "scene_id": args.scene, "model": MODEL,
                            "model_revision": revision, "keypoint_names": list(COCO),
                            "manifest_sha256": sha256(args.manifest), "observations": rows})
    confident = [np.mean(np.asarray(r["scores"]) >= 0.3) for r in rows]
    print(f"{len(rows)} quan sát, trung bình {np.mean(confident) * 100:.0f}% keypoint có độ tin cậy >= 0.3"
          if rows else "Không có quan sát người đi bộ nào")
    print(args.output)


if __name__ == "__main__":
    main()
