"""Bring published OmniRe PandaSet poses into the pinned instance-ID convention."""
import argparse
import math
import re
import shutil
import zipfile
from pathlib import Path

from .prepare import dump_json, read_json, sha256

POSE_ARCHIVE = ("https://drive.usercontent.google.com/download?"
                "id=1ODzoH7SxNzjOThhKUc_n2LLXcOAeGMCM&export=download&confirm=t")
POSE_ARCHIVE_SHA256 = "4dbedde09a9096787731b9d885913b917bd720ea13b9c76b965e5c0b7820695f"

# Preserve the classes and the insertion order of the pinned PandaSetProcessor.
CLASSES = {
    "Pedestrian", "Pedestrian with Object", "Bicycle", "Animals - Other",
    "Bus", "Car", "Emergency Vehicle", "Medium-sized Truck", "Motorcycle", "Motorized Scooter",
    "Other Vehicle - Construction Vehicle", "Other Vehicle - Pedicab", "Other Vehicle - Uncommon",
    "Personal Mobility Device", "Pickup Truck", "Semi-truck", "Train", "Tram / Subway",
}


def instances(raw):
    import pandas as pd

    count = len(read_json(raw / "camera" / "front_camera" / "timestamps.json"))
    objects = {}
    for frame in range(count):
        annotations = pd.read_pickle(raw / "annotations" / "cuboids" / f"{frame:02d}.pkl.gz")
        for row in annotations.to_dict("records"):
            uuid, label = str(row["uuid"]), row["label"]
            if label not in CLASSES:
                continue
            item = objects.setdefault(uuid, {
                "id": uuid, "class_name": label, "sibling_id": str(row["cuboids.sibling_id"]),
                "frame_annotations": {"frame_idx": [], "obj_to_world": [], "box_size": [], "stationary": []},
            })["frame_annotations"]
            cos, sin = math.cos(row["yaw"]), math.sin(row["yaw"])
            x, y, z = (float(row[f"position.{axis}"]) for axis in "xyz")
            item["frame_idx"].append(frame)
            item["obj_to_world"].append([[cos, -sin, 0, x], [sin, cos, 0, y],
                                        [0, 0, 1, z], [0, 0, 0, 1]])
            item["box_size"].append([float(row[f"dimensions.{axis}"]) for axis in "xyz"])
            item["stationary"].append(bool(row["stationary"]))
    pairs, seen = [], set()
    for uuid, item in objects.items():
        sibling = item["sibling_id"]
        pair = frozenset((uuid, sibling))
        if sibling != "-" and pair not in seen:
            seen.add(pair)
            pairs.append((uuid, sibling))
    for a, b in pairs:
        if a in objects and b in objects:
            remove = b if len(objects[a]["frame_annotations"]["frame_idx"]) > len(
                objects[b]["frame_annotations"]["frame_idx"]
            ) else a
            del objects[remove]
    dynamic = (item for item in objects.values() if not all(item["frame_annotations"]["stationary"]))
    return {str(index): item for index, item in enumerate(dynamic)}


def main():
    import requests

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True, help="Cache the official pose ZIP here")
    parser.add_argument("--scenes", nargs="+", required=True)
    args = parser.parse_args()
    if not all(re.fullmatch(r"[0-9]{3}", scene) for scene in args.scenes):
        parser.error("Use three-digit PandaSet scene IDs")
    if not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.archive.with_suffix(".part")
        with requests.get(POSE_ARCHIVE, timeout=(15, 120), stream=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024**2):
                    handle.write(chunk)
        if sha256(temporary) != POSE_ARCHIVE_SHA256:
            raise ValueError("The upstream OmniRe zip checksum changed")
        temporary.replace(args.archive)
    if sha256(args.archive) != POSE_ARCHIVE_SHA256:
        raise ValueError("The cached OmniRe zip checksum differs from the published bundle")
    with zipfile.ZipFile(args.archive) as archive:
        for scene in args.scenes:
            member = f"pandaset/processed/{scene}/humanpose/smpl.pkl"
            if member not in archive.namelist():
                raise ValueError(f"OmniRe's pose bundle lacks {scene}. Use scripts/omnire-pose.sh.")
            folder = args.processed / scene
            info = instances(args.raw / scene)
            dump_json(folder / "instances" / "instances_info.json", info)
            path = folder / "humanpose" / "smpl.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".part")
            with archive.open(member) as source, temp.open("wb") as target:
                shutil.copyfileobj(source, target)
            temp.replace(path)
            print(f"{scene}: {len(info)} OmniRe instance IDs")


if __name__ == "__main__":
    main()
