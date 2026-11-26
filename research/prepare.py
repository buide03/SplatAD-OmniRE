"""Export SMPL tracks and freeze the observations used by every comparison."""
import argparse
import hashlib
import itertools
import json
import math
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from .sync_video import FRONT_CAMERAS

CAMERAS = ("front", "front_left", "front_right", "left", "right", "back")
OMNIRE_POSE_CAMERAS = CAMERAS
PEDESTRIANS = {"Pedestrian", "Pedestrian with Object"}
CORNERS = np.array(list(itertools.product((-0.5, 0.5), repeat=3)))


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def front_pose_summary(manifest):
    """Count selected SMPL tracks and valid person-frames seen by the front cameras."""
    selected = set(manifest.get("selected_uuids", []))
    frames = {"4d_humans": set(), "omnire_interpolation": set()}
    tracks = set()
    reference = set()
    for row in manifest["observations"]:
        source = row.get("pose_source")
        if (row["uuid"] in selected and row["camera"] in FRONT_CAMERAS
                and row.get("smpl_valid") and source in frames):
            observed = (row["uuid"], row["frame"])
            if row.get("pose_origin") == "omnire_reference":
                reference.add(observed)
            else:
                frames[source].add(observed)
            tracks.add(row["uuid"])
    summary = {"tracks": len(tracks), "direct_frames": len(frames["4d_humans"]),
               "completed_frames": len(frames["omnire_interpolation"])}
    if reference:
        summary["reference_frames"] = len(reference)
    return summary


def camera_at(scene, camera, frame):
    from scipy.spatial.transform import Rotation

    folder = scene / "camera" / f"{camera}_camera"
    calibration, poses, times = _camera_metadata(folder)
    p = poses[frame]
    q = [p["heading"][axis] for axis in "xyzw"]
    t = np.array([p["position"][axis] for axis in "xyz"])
    return Rotation.from_quat(q).as_matrix(), t, calibration, float(times[frame])


@lru_cache(maxsize=48)
def _camera_metadata(folder):
    return tuple(read_json(folder / name) for name in ("intrinsics.json", "poses.json", "timestamps.json"))


def projected_box(center, dimensions, yaw, camera, height, width):
    rot, origin, k, _ = camera
    cos, sin = math.cos(yaw), math.sin(yaw)
    local_to_world = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    xyz = ((CORNERS * dimensions) @ local_to_world.T + center - origin) @ rot
    if (xyz[:, 2] <= 0.5).any():
        return None
    pixel = xyz[:, :2] / xyz[:, 2:] * [k["fx"], k["fy"]] + [k["cx"], k["cy"]]
    lower = np.maximum(np.floor(pixel.min(0)), 0).astype(int)
    upper = np.minimum(np.ceil(pixel.max(0)), [width, height]).astype(int)
    return [*lower.tolist(), *upper.tolist()] if np.all(upper > lower) else None


def box_records(scene, frame, labels=PEDESTRIANS):
    import pandas as pd

    data = pd.read_pickle(scene / "annotations" / "cuboids" / f"{frame:02d}.pkl.gz")
    for row in data.to_dict("records"):
        if row["label"] in labels and row["cuboids.sensor_id"] != 1:
            yield row


def bin_for(distance):
    if distance <= 15:
        return "near"
    return "far" if distance >= 30 else "mid"


def validate_tracks(document, scene):
    from .geometry import JOINTS

    expected = (1, str(scene), "pandaset_world", "wxyz", "m", "s",
                "pandaset_camera_epoch", list(JOINTS),
                "cuboid_center_minus_posed_mesh_bbox_center")
    fields = ("schema_version", "scene_id", "coordinate_system", "quaternion_order", "length_unit",
              "time_unit", "time_reference", "joint_order", "translation_definition")
    if (tuple(document.get(key) for key in fields) != expected
            or document.get("smpl_version") not in ("smpl_neutral_v1.0", "smpl_neutral_v1.1")):
        raise ValueError("Scene, model or coordinate conventions in SMPL JSON are incompatible")
    uuids = set()
    for track in document["tracks"]:
        uuid = track["uuid"]
        if not isinstance(uuid, str) or not uuid or uuid in uuids:
            raise ValueError(f"Duplicate or invalid UUID: {uuid}")
        uuids.add(uuid)
        betas = np.asarray(track["betas"])
        if betas.shape != (10,) or not np.isfinite(betas).all():
            raise ValueError(f"Invalid betas: {uuid}")
        previous = (-1, -math.inf)
        for frame in track["frames"]:
            index, timestamp = frame["frame_idx"], frame["timestamp_s"]
            if index <= previous[0] or not math.isfinite(timestamp) or timestamp <= previous[1]:
                raise ValueError(f"Non-monotonic frames: {uuid}")
            previous = index, timestamp
            if not frame["valid"]:
                continue
            q = np.asarray(frame["root_quat_wxyz"])
            body = np.asarray(frame["body_quats_wxyz"])
            t = np.asarray(frame["root_translation"])
            if (q.shape != (4,) or body.shape != (23, 4) or t.shape != (3,)
                    or not np.isfinite(t).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-3)
                    or not np.allclose(np.linalg.norm(body, axis=-1), 1, atol=1e-3)):
                raise ValueError(f"Invalid pose: {uuid} frame {index}")
    return document


def prepare_smpl(args):
    # SMPL v1.0 and v1.1 can contain Chumpy objects written before these aliases were removed.
    import builtins
    import inspect

    aliases = ("bool", "int", "float", "complex", "object", "str")
    added = {name: getattr(builtins, name) for name in aliases if name not in vars(np)}
    if "unicode" not in vars(np):
        added["unicode"] = str
    vars(np).update(added)
    legacy_inspect = not hasattr(inspect, "getargspec")
    if legacy_inspect:
        inspect.getargspec = inspect.getfullargspec
    try:
        with args.input.open("rb") as handle:
            source = pickle.load(handle, encoding="latin1")
    finally:
        for name in added:
            delattr(np, name)
        if legacy_inspect:
            del inspect.getargspec
    tree = source["kintree_table"]
    ids = {int(joint): index for index, joint in enumerate(tree[1])}
    parents = [-1] + [ids[int(joint)] for joint in tree[0, 1:]]
    names = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "f")
    arrays = {}
    for name in names:
        value = source[name]
        arrays[name] = np.asarray(value.toarray() if hasattr(value, "toarray") else value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, parents=parents, smpl_version=args.model_version)


def export_tracks(args):
    import joblib
    import torch
    from scipy.spatial.transform import Rotation

    from .geometry import JOINTS, BodyModel, apply, blend, joint_transforms

    scene = args.raw / args.scene
    processed = args.processed / args.scene
    instances = read_json(processed / "instances" / "instances_info.json")
    poses = joblib.load(processed / "humanpose" / "smpl.pkl")
    body_model = BodyModel(args.body)
    frames_total = len(_camera_metadata(scene / "camera" / "front_camera")[2])
    canonical_ids = {str(row["uuid"]) for f in range(frames_total) for row in box_records(scene, f)}
    document = {
        "schema_version": 1, "scene_id": args.scene, "coordinate_system": "pandaset_world",
        "quaternion_order": "wxyz", "length_unit": "m", "time_unit": "s",
        "time_reference": "pandaset_camera_epoch", "smpl_version": body_model.version,
        "joint_order": list(JOINTS), "translation_definition": "cuboid_center_minus_posed_mesh_bbox_center",
        "body_sha256": sha256(args.body), "tracks": [],
    }
    for identity, prediction in poses.items():
        instance = instances[str(identity)]
        if instance["class_name"] not in PEDESTRIANS:
            continue
        uuid = str(instance["id"])
        if uuid not in canonical_ids:
            uuid = str(instance.get("sibling_id"))
        if uuid not in canonical_ids:
            continue
        annotations = instance["frame_annotations"]
        cuboids = dict(zip(annotations["frame_idx"], annotations["obj_to_world"], strict=True))
        candidates = {}
        for f, valid in enumerate(np.asarray(prediction["valid_mask"], dtype=bool)):
            cam = int(prediction["selected_cam_idx"][f])
            if not valid or f not in cuboids or not 0 <= cam < len(OMNIRE_POSE_CAMERAS):
                continue
            smpl = prediction["smpl"]
            betas = np.asarray(smpl["betas"][f]).reshape(10)
            matrices = np.concatenate((np.asarray(smpl["global_orient"][f]).reshape(1, 3, 3),
                                       np.asarray(smpl["body_pose"][f]).reshape(23, 3, 3)))
            if not np.isfinite(betas).all() or not np.isfinite(matrices).all() or (np.linalg.det(matrices) < 0.5).any():
                continue
            candidates[f] = (OMNIRE_POSE_CAMERAS[cam], betas, matrices)
        if not candidates:
            continue
        matched = np.asarray(prediction.get("direct_4d_mask", prediction.get("matched_mask", prediction["valid_mask"])),
                             dtype=bool)
        betas = np.median([item[1] for f, item in candidates.items() if matched[f]]
                          or [item[1] for item in candidates.values()], axis=0)
        vertices, joints, inverse = body_model.canonical(torch.tensor(betas, dtype=torch.float32))
        frames = []
        for f in sorted(cuboids):
            default_time = camera_at(scene, "front", f)[3] - 0.01
            frame = {"frame_idx": f, "timestamp_s": default_time, "source_cam": None, "valid": False,
                     "pose_source": None, "root_translation": None, "root_quat_wxyz": None, "body_quats_wxyz": None}
            if f in candidates:
                camera, _, matrices = candidates[f]
                rot, _, _, time = camera_at(scene, camera, f)
                matrices = matrices.copy()
                matrices[0] = rot @ matrices[0]
                pose = Rotation.from_matrix(matrices)
                transform = joint_transforms(torch.tensor(pose.as_matrix(), dtype=torch.float32),
                                             joints, body_model.parents) @ inverse
                points = apply(blend(body_model.weights, transform), vertices)
                shift = (points.amin(0) + points.amax(0)) / 2
                translation = np.asarray(cuboids[f])[:3, 3] - shift.numpy()
                q = pose.as_quat()[:, [3, 0, 1, 2]]
                frame.update(
                    timestamp_s=time - 0.01, source_cam=camera, valid=True,
                    pose_source="4d_humans" if matched[f] else "omnire_interpolation",
                    root_translation=translation.tolist(), root_quat_wxyz=q[0].tolist(),
                    body_quats_wxyz=q[1:].tolist(),
                )
            frames.append(frame)
        track = {"uuid": uuid, "betas": betas.tolist(), "frames": frames}
        old = next((t for t in document["tracks"] if t["uuid"] == uuid), None)
        if old is None or sum(f["valid"] for f in old["frames"]) < len(candidates):
            if old is not None:
                document["tracks"].remove(old)
            document["tracks"].append(track)
    document["tracks"].sort(key=lambda track: track["uuid"])
    validate_tracks(document, args.scene)
    dump_json(args.output, document)
    print(f"{args.scene}: exported {len(document['tracks'])} tracks to {args.output}")


def inventory(args):
    import pandas as pd

    raw_scene = args.raw / args.scene
    tracks = validate_tracks(read_json(args.tracks), args.scene)["tracks"] if args.tracks else []
    usable = {(track["uuid"], frame["frame_idx"]): frame
              for track in tracks for frame in track["frames"] if frame["valid"]}
    bbox_review = {}
    if args.review:
        bbox_review = {(r["uuid"], r["camera"], int(r["frame"])): r
                       for r in read_json(args.review)["observations"]}
    times = read_json(raw_scene / "lidar" / "timestamps.json")
    rows = []
    for f in range(len(times)):
        cuboids = list(box_records(raw_scene, f))
        if not cuboids:
            continue
        pc = pd.read_pickle(raw_scene / "lidar" / f"{f:02d}.pkl.gz")
        points = pc.loc[pc["d"] == 0, ["x", "y", "z"]].to_numpy()
        for box in cuboids:
            uuid = str(box["uuid"])
            center = np.array([box[f"position.{axis}"] for axis in "xyz"])
            size = np.array([box[f"dimensions.{axis}"] for axis in "xyz"])
            yaw = float(box["yaw"])
            cos, sin = math.cos(yaw), math.sin(yaw)
            rot = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
            hits = int((np.abs((points - center) @ rot) <= size / 2).all(1).sum())
            for name in args.cameras:
                camera = camera_at(raw_scene, name, f)
                h, w = (820 if name == "back" else 1080), 1920
                projected = projected_box(center, size, yaw, camera, h, w)
                review = bbox_review.get((uuid, name, f), {})
                bbox = review.get("bbox_xyxy", projected)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    raise ValueError(f"Invalid reviewed bbox for {uuid}, {name}, frame {f}")
                distance = float(np.linalg.norm(center - camera[1]))
                visible = review.get("limbs_visible")
                occlusion = review.get("occlusion")
                silhouette = review.get("silhouette_xy")
                if silhouette is not None:
                    polygon = np.asarray(silhouette, dtype=float)
                    if (polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2
                            or not np.isfinite(polygon).all() or (polygon < 0).any()
                            or (polygon >= [w, h]).any()):
                        raise ValueError(f"Invalid silhouette for {uuid}, {name}, frame {f}")
                pose = usable.get((uuid, f), {})
                pose_source = pose.get("pose_source")
                rows.append({
                    "scene_id": args.scene, "uuid": uuid, "camera": name, "frame": f, "scan": f,
                    "timestamp_s": camera[3], "lidar_timestamp_s": times[f], "distance_m": distance,
                    "bin": bin_for(distance), "bbox_xyxy": bbox, "bbox_height_px": y2 - y1,
                    "bbox_source": "reviewed" if "bbox_xyxy" in review else "projected_cuboid",
                    "limbs_visible": visible, "occlusion": occlusion, "raw_lidar_hits": hits,
                    "smpl_valid": pose_source is not None, "pose_source": pose_source,
                    "pose_origin": pose.get("pose_origin"),
                    "box_center": center.tolist(), "box_size": size.tolist(), "box_yaw": yaw,
                    "usable": visible is not False and (visible is True or y2 - y1 >= args.min_height),
                    "stationary": bool(box["stationary"]),
                    "keypoints": review.get("keypoints", {}),
                    "silhouette_xy": silhouette,
                    "pose_monitor": bool(review.get("pose_monitor", False)),
                })
    by_frame = {}
    for row in rows:
        if row["usable"]:
            by_frame.setdefault((row["camera"], row["frame"]), []).append(row)
    probe_keys = set()
    for camera in args.cameras:
        options = [(frame, group) for (name, frame), group in sorted(by_frame.items()) if name == camera]
        if not options:
            continue
        # Pick across the motion span. Prefer simultaneous pedestrians, available pose, and returned rays.
        for segment in np.array_split(np.arange(len(options)), min(len(options), args.probe_frames)):
            choice = max(segment, key=lambda i: (
                sum(r["limbs_visible"] is True for r in options[i][1]),
                len({r["bin"] for r in options[i][1]}), len(options[i][1]),
                sum(r["smpl_valid"] for r in options[i][1]), sum(r["raw_lidar_hits"] > 0 for r in options[i][1]),
            ))
            probe_keys.add((camera, options[choice][0]))
    for row in rows:
        row["probe"] = row["usable"] and (row["pose_monitor"] or (row["camera"], row["frame"]) in probe_keys)
    long_enough = {track["uuid"] for track in tracks if sum(frame["valid"] for frame in track["frames"]) >= 2}
    tracks_selected = sorted({row["uuid"] for row in rows if row["uuid"] in long_enough and row["usable"] and row["smpl_valid"]
                              and not row["stationary"]})
    if args.uuids:
        missing = set(args.uuids) - set(tracks_selected)
        if missing:
            raise ValueError(f"Requested UUIDs lack a usable moving SMPL observation: {sorted(missing)}")
        tracks_selected = sorted(set(args.uuids))
    counts = {}
    for name in ("near", "mid", "far"):
        group = [row for row in rows if row["bin"] == name]
        visible = [row for row in group if row["usable"]]
        samples = {(r["uuid"], r["scan"]): r["raw_lidar_hits"] for r in visible}
        counts[name] = {
            "observations": len(group), "usable": len(visible),
            "n_uuid": len({r["uuid"] for r in group}),
            "n_uuid_smpl": len({r["uuid"] for r in visible if r["smpl_valid"]}),
            "track_frames_smpl": len({(r["uuid"], r["frame"]) for r in visible if r["smpl_valid"]}),
            "track_frames_4d_humans": len({(r["uuid"], r["frame"]) for r in visible if r["pose_source"] == "4d_humans"}),
            "manually_visible_limbs": sum(r["limbs_visible"] is True for r in visible),
            "raw_lidar_hits": sum(samples.values()),
            "max_simultaneous": max((sum(r["bin"] == name for r in g) for g in by_frame.values()), default=0),
            "max_simultaneous_smpl": max((sum(r["bin"] == name and r["smpl_valid"] for r in g)
                                          for g in by_frame.values()), default=0),
        }
    dump_json(args.output, {
        "schema_version": 1, "scene_id": args.scene, "cameras": args.cameras, "near_max_m": 15,
        "far_min_m": 30, "min_bbox_height_px": args.min_height, "reviewed": bool(args.review),
        "tracks_sha256": sha256(args.tracks) if args.tracks else None, "selected_uuids": tracks_selected,
        "foot_contacts": read_json(args.review).get("foot_contacts", []) if args.review else [],
        "coverage": counts, "observations": rows,
    })
    print(json.dumps(counts, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    prepare = commands.add_parser("prepare-smpl")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--model-version", choices=("smpl_neutral_v1.0", "smpl_neutral_v1.1"),
                         default="smpl_neutral_v1.1")
    prepare.set_defaults(action=prepare_smpl)
    export = commands.add_parser("export-tracks")
    export.add_argument("--processed", type=Path, required=True)
    export.add_argument("--body", type=Path, required=True)
    export.set_defaults(action=export_tracks)
    scan = commands.add_parser("inventory")
    scan.add_argument("--tracks", type=Path)
    scan.add_argument("--review", type=Path)
    scan.add_argument("--uuids", nargs="+")
    scan.add_argument("--cameras", nargs="+", choices=CAMERAS, default=["front_right"])
    scan.add_argument("--min-height", type=int, default=28)
    scan.add_argument("--probe-frames", type=int, default=8)
    scan.set_defaults(action=inventory)
    for subparser in (export, scan):
        subparser.add_argument("--raw", type=Path, required=True)
        subparser.add_argument("--scene", required=True)
        subparser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action is inventory and (args.min_height < 1 or args.probe_frames < 1):
        parser.error("Observation height and probe count must be positive")
    args.action(args)


if __name__ == "__main__":
    main()
