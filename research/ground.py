"""Estimate local support planes from PandaSet scans for near-range SMPL tracks."""

import argparse
from pathlib import Path

import numpy as np

from .prepare import dump_json, read_json, sha256


def fit_road_plane(points, center, bottom):
    """Fit an upward plane near the cuboid bottom, with a residual/inlier gate."""
    if len(points) < 16:
        return None
    xy = points[:, :2] - center[:2]
    z = points[:, 2] - bottom
    features = np.column_stack((xy, np.ones(len(points))))
    keep = np.ones(len(z), dtype=bool)
    for _ in range(3):
        coefficients, _, rank, _ = np.linalg.lstsq(features[keep], z[keep], rcond=None)
        if rank < 3:
            return None
        residual = z - features @ coefficients
        keep = np.abs(residual - np.median(residual[keep])) <= 0.08
        if keep.sum() < max(16, 0.55 * len(z)):
            return None
    a, b, c = np.linalg.lstsq(features[keep], z[keep], rcond=None)[0]
    if np.hypot(a, b) > 0.35 or abs(c) > 0.30:
        return None
    # n·p + d = 0. The normal points up and signed distances are in meters.
    plane = np.array([-a, -b, 1., a * center[0] + b * center[1] - c - bottom])
    return (plane / np.linalg.norm(plane[:3])).tolist()


def build_ground(raw, manifest_path):
    import pandas as pd

    manifest = read_json(manifest_path)
    scene = raw / manifest["scene_id"]
    selected = set(manifest["selected_uuids"])
    by_frame = {}
    for row in manifest["observations"]:
        if row["uuid"] in selected and row["bin"] == "near" and row["usable"] and row["smpl_valid"]:
            by_frame.setdefault(row["frame"], {})[row["uuid"]] = row
    planes = {}
    for frame, actors in by_frame.items():
        cloud = pd.read_pickle(scene / "lidar" / f"{frame:02d}.pkl.gz")
        points = cloud.loc[cloud["d"] == 0, ["x", "y", "z"]].to_numpy()
        boxes = pd.read_pickle(scene / "annotations" / "cuboids" / f"{frame:02d}.pkl.gz")
        boxes = boxes.loc[boxes["cuboids.sensor_id"] != 1].to_dict("records")
        for uuid, row in actors.items():
            center = np.asarray(row["box_center"])
            bottom = center[2] - row["box_size"][2] / 2
            dist = np.linalg.norm(points[:, :2] - center[:2], axis=1)
            nearby = points[(dist >= 0.30) & (dist <= 2.0) & (np.abs(points[:, 2] - bottom) <= 0.40)]
            for box in boxes:
                position = np.array([box[f"position.{a}"] for a in "xyz"])
                size = np.array([box[f"dimensions.{a}"] for a in "xyz"])
                if np.linalg.norm(position[:2] - center[:2]) > np.linalg.norm(size[:2]) / 2 + 2.1:
                    continue
                cosine, sine = np.cos(box["yaw"]), np.sin(box["yaw"])
                rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
                inside = (np.abs((nearby - position) @ rotation) <= size / 2 + 0.05).all(1)
                nearby = nearby[~inside]
            plane = fit_road_plane(nearby, center, bottom)
            if plane is not None:
                planes.setdefault(uuid, {})[str(frame)] = plane
    return {"schema_version": 1, "scene_id": manifest["scene_id"],
            "manifest_sha256": sha256(manifest_path), "planes": planes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_ground(args.raw, args.manifest)
    dump_json(args.output, result)
    print(f"{result['scene_id']}: fitted {sum(map(len, result['planes'].values()))} local road planes")


if __name__ == "__main__":
    main()
