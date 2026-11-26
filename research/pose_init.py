"""Compare local 4D-Humans with OmniRe and optionally fill missing initialized poses."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from .prepare import dump_json, read_json, sha256, validate_tracks


REFERENCE_MODES = ("off", "compare", "fill")


def angular_difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    dot = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(2 * np.arccos(np.clip(np.abs(dot), 0, 1)))


def posed_center(body, canonical, frame):
    import torch

    from .geometry import apply, blend, joint_transforms, quat_matrix

    vertices, joints, inverse = canonical
    rotations = torch.tensor([frame["root_quat_wxyz"], *frame["body_quats_wxyz"]], dtype=torch.float32)
    transform = joint_transforms(quat_matrix(rotations), joints, body.parents) @ inverse
    points = apply(blend(body.weights, transform), vertices)
    return (points.amin(0) + points.amax(0)) / 2


def merge_initialization(primary, reference, mode, body=None):
    scene = primary["scene_id"]
    combined = deepcopy(validate_tracks(primary, scene))
    if mode not in REFERENCE_MODES or (mode == "off" and reference is not None):
        raise ValueError("Invalid OmniRe reference mode")
    if reference is not None:
        validate_tracks(reference, scene)
        if any(reference[key] != primary[key] for key in ("smpl_version", "body_sha256")):
            raise ValueError("Local and reference poses must use the same SMPL body model")
    reference_tracks = {track["uuid"]: track for track in (reference or {}).get("tracks", [])}
    counts = {
        "primary_tracks": len(combined["tracks"]), "reference_tracks": len(reference_tracks), "shared_tracks": 0,
        "primary_valid_frames": 0, "overlap_frames": 0, "available_fills": 0, "filled_frames": 0,
        "timestamp_conflicts": 0,
    }
    differences = {"root_translation_m": [], "root_rotation_deg": [], "mean_joint_rotation_deg": [],
                   "timestamp_difference_ms": []}
    filled = []
    for track in combined["tracks"]:
        uuid = track["uuid"]
        frames = track["frames"]
        other = reference_tracks.get(uuid) if any(frame["valid"] for frame in frames) else None
        counts["shared_tracks"] += bool(other)
        reference_frames = {frame["frame_idx"]: frame for frame in (other or {}).get("frames", []) if frame["valid"]}
        shapes = None
        for index, frame in enumerate(frames):
            frame["pose_origin"] = "local_4d_humans" if frame["valid"] else None
            candidate = reference_frames.get(frame["frame_idx"])
            if frame["valid"]:
                counts["primary_valid_frames"] += 1
                if candidate:
                    counts["overlap_frames"] += 1
                    differences["root_translation_m"].append(np.linalg.norm(
                        np.subtract(frame["root_translation"], candidate["root_translation"])))
                    differences["root_rotation_deg"].append(
                        angular_difference(frame["root_quat_wxyz"], candidate["root_quat_wxyz"]))
                    differences["mean_joint_rotation_deg"].append(angular_difference(
                        frame["body_quats_wxyz"], candidate["body_quats_wxyz"]).mean())
                    differences["timestamp_difference_ms"].append(
                        abs(frame["timestamp_s"] - candidate["timestamp_s"]) * 1000)
                continue
            if not candidate:
                continue
            previous = frames[index - 1]["timestamp_s"] if index else -np.inf
            following = frames[index + 1]["timestamp_s"] if index + 1 < len(frames) else np.inf
            if not previous < candidate["timestamp_s"] < following:
                counts["timestamp_conflicts"] += 1
                continue
            counts["available_fills"] += 1
            if mode != "fill":
                continue
            replacement = deepcopy(candidate)
            replacement["pose_origin"] = "omnire_reference"
            if track["betas"] != other["betas"]:
                import torch

                if body is None:
                    raise ValueError("Filling a pose with a different shape requires the SMPL body model")
                with torch.no_grad():
                    if shapes is None:
                        shapes = [body.canonical(torch.tensor(t["betas"], dtype=torch.float32)) for t in (track, other)]
                    # Preserve the reference cuboid center with the local person's shape.
                    delta = posed_center(body, shapes[1], candidate) - posed_center(body, shapes[0], candidate)
                    replacement["root_translation"] = (np.asarray(candidate["root_translation"]) + delta.numpy()).tolist()
            frames[index] = replacement
            filled.append({"uuid": uuid, "frame": frame["frame_idx"], "camera": replacement["source_cam"]})
            counts["filled_frames"] += 1
    if not counts["primary_valid_frames"]:
        raise ValueError(f"{scene}: local 4D-Humans provided no valid pose; inspect its extraction output")
    result = {
        "scene_id": scene, "primary": "local_4d_humans", "reference_mode": mode,
        "reference_available": reference is not None, "counts": counts,
        "median_difference": {key: float(np.median(values)) if values else None for key, values in differences.items()},
        "filled_frames": filled,
    }
    combined["initialization"] = {"primary": result["primary"], "reference_mode": mode,
                                  "reference_available": result["reference_available"]}
    return validate_tracks(combined, scene), result


def main():
    from .geometry import BodyModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--mode", choices=REFERENCE_MODES, default="fill")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    primary = read_json(args.primary)
    if sha256(args.body) != primary["body_sha256"]:
        parser.error("Use the same SMPL body used to export the local 4D-Humans tracks")
    reference = read_json(args.reference) if args.reference else None
    body = BodyModel(args.body) if args.mode == "fill" and reference is not None else None
    combined, report = merge_initialization(primary, reference, args.mode, body)
    report["primary_sha256"] = sha256(args.primary)
    report["reference_sha256"] = sha256(args.reference) if args.reference else None
    dump_json(args.report, report)
    dump_json(args.output, combined)
    print(f"{report['scene_id']}: valid local and OmniRe frames: {report['counts']}")


if __name__ == "__main__":
    main()
