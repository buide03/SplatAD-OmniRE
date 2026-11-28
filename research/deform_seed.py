"""Seed R3 deformable pedestrians from PandaSet: LiDAR returns and camera colours in the cuboid frame.

SplatAD seeds every actor with 500 random points in its box plus the LiDAR returns inside it, mirrors
and duplicates them, and colours all of them at random because PandaSet LiDAR has no RGB. For a far
pedestrian most rows are that random filler. This step builds the seed from the data instead, in the
cuboid frame SplatAD also uses (PandaSet yaw rotation, dimensions x, y, z):

* every Pandar64 return inside the cuboid over the whole sequence, coloured by the cameras that see it
  unoccluded at the same frame (median over views);
* fill points in an upright cylinder inside the box, kept only where a camera sees this person: a
  LiDAR return of the same person falls in the same image cell at about the same depth, and nothing
  closer covers it. Their colour is the median over those views.

The model reads the result with ``--deform-seed`` and replaces the bank rows of each deformable
pedestrian that has enough points. Others keep SplatAD's seed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .prepare import CAMERAS, PEDESTRIANS, camera_at, read_json

SCHEMA_VERSION = 1
CELL_PX = 8
OCCLUSION_M = 0.3
BODY_DEPTH_M = (-0.2, 0.8)


def yaw_matrix(yaw):
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def pedestrian_boxes(scene, frame):
    """Moving pedestrian cuboids of the 360 LiDAR at one frame: uuid -> (center, size, rotation)."""
    import pandas as pd

    rows = pd.read_pickle(scene / "annotations" / "cuboids" / f"{frame:02d}.pkl.gz").to_dict("records")
    boxes = {}
    for row in rows:
        if row["label"] not in PEDESTRIANS or row["stationary"] or row["cuboids.sensor_id"] == 1:
            continue
        center = np.array([row[f"position.{a}"] for a in "xyz"], dtype=np.float64)
        size = np.array([row[f"dimensions.{a}"] for a in "xyz"], dtype=np.float64)
        boxes[row["uuid"]] = (center, size, yaw_matrix(row["yaw"]))
    return boxes


def cylinder(count, size, generator):
    """Uniform points in the upright elliptic cylinder inscribed in a box of this size."""
    radius = np.sqrt(generator.random(count))
    angle = generator.random(count) * 2 * np.pi
    return np.column_stack((radius * np.cos(angle) * size[0] / 2, radius * np.sin(angle) * size[1] / 2,
                            (generator.random(count) - 0.5) * size[2]))


class View:
    """One camera at one frame: projection, nearest LiDAR depth per image cell, and the image."""

    def __init__(self, scene, camera, frame, cloud):
        from PIL import Image

        self.rotation, self.origin, k, _ = camera_at(scene, camera, frame)
        self.k = np.array([k["fx"], k["fy"]]), np.array([k["cx"], k["cy"]])
        with Image.open(scene / "camera" / f"{camera}_camera" / f"{frame:02d}.jpg") as image:
            self.image = np.asarray(image.convert("RGB"))
        height, width = self.image.shape[:2]
        self.grid = (height + CELL_PX - 1) // CELL_PX, (width + CELL_PX - 1) // CELL_PX
        self.nearest = self.depth_grid(cloud)
        self.nearest_around = dilate_min(self.nearest)

    def project(self, points):
        """Pixel, depth and a mask for points in front of the camera and inside the image."""
        local = (points - self.origin) @ self.rotation
        depth = local[:, 2]
        pixel = local[:, :2] / np.maximum(depth, 1e-6)[:, None] * self.k[0] + self.k[1]
        height, width = self.image.shape[:2]
        inside = (depth > 0.5) & (pixel[:, 0] >= 0) & (pixel[:, 0] < width) & (pixel[:, 1] >= 0) & (pixel[:, 1] < height)
        return pixel, depth, inside

    def cells(self, pixel):
        return (pixel[:, 1] // CELL_PX).astype(np.int64), (pixel[:, 0] // CELL_PX).astype(np.int64)

    def depth_grid(self, points):
        grid = np.full(self.grid, np.inf)
        pixel, depth, inside = self.project(points)
        rows, cols = self.cells(pixel[inside])
        np.minimum.at(grid, (rows, cols), depth[inside])
        return grid

    def colours(self, pixel):
        return self.image[pixel[:, 1].astype(np.int64), pixel[:, 0].astype(np.int64)]


def dilate_min(grid):
    padded = np.pad(grid, 1, constant_values=np.inf)
    rows, cols = grid.shape
    return np.min([padded[i:i + rows, j:j + cols] for i in range(3) for j in range(3)], axis=0)


def median_by(owner, values, count):
    """Per-owner median of colour observations; owners without one get -1."""
    result = np.full((count, 3), -1.0)
    if not len(owner):
        return result
    order = np.argsort(owner, kind="stable")
    owner, values = owner[order], values[order]
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    for start, end in zip(starts, np.r_[starts[1:], len(owner)]):
        result[owner[start]] = np.median(values[start:end], axis=0)
    return result


def build_seed(raw, scene_id, cameras=CAMERAS, target=2000, max_points=5000, candidates=4, min_views=2, seed=0):
    import pandas as pd

    scene = raw / scene_id
    frames = len(read_json(scene / "lidar" / "timestamps.json"))
    generator = np.random.default_rng(seed)
    boxes = [pedestrian_boxes(scene, frame) for frame in range(frames)]
    sizes = {}
    for frame_boxes in boxes:
        for uuid, (_, size, _) in frame_boxes.items():
            sizes[uuid] = np.maximum(sizes.get(uuid, size), size)
    # Candidate fill points per pedestrian, fixed in its box frame for the whole sequence.
    fill = {uuid: cylinder(candidates * target, size, generator) for uuid, size in sizes.items()}
    fill_seen = {uuid: ([], []) for uuid in sizes}
    lidar = {uuid: ([], []) for uuid in sizes}  # box-frame points and their colour observations
    for frame, frame_boxes in enumerate(boxes):
        if not frame_boxes:
            continue
        cloud = pd.read_pickle(scene / "lidar" / f"{frame:02d}.pkl.gz")
        world = cloud.loc[cloud["d"] == 0, ["x", "y", "z"]].to_numpy(dtype=np.float64)
        members = {}
        for uuid, (center, size, rotation) in frame_boxes.items():
            local = (world - center) @ rotation
            inside = (np.abs(local) <= size / 2 + 0.05).all(axis=1)
            members[uuid] = (world[inside], local[inside])
        views = [View(scene, camera, frame, world) for camera in cameras]
        for uuid, (center, size, rotation) in frame_boxes.items():
            hits, local = members[uuid]
            colour_owner, colour_value = [], []
            person_depth = []
            for view in views:
                pixel, depth, inside = view.project(hits)
                rows, cols = view.cells(pixel[inside])
                visible = depth[inside] - view.nearest[rows, cols] <= OCCLUSION_M
                index = np.flatnonzero(inside)[visible]
                colour_owner.append(index)
                colour_value.append(view.colours(pixel[index]))
                # This person's own returns mark where the camera sees them, per image cell.
                grid = np.full(view.grid, np.inf)
                np.minimum.at(grid, (rows[visible], cols[visible]), depth[inside][visible])
                person_depth.append(dilate_min(grid))
            colour = median_by(np.concatenate(colour_owner), np.concatenate(colour_value).astype(np.float64), len(hits))
            lidar[uuid][0].append(local)
            lidar[uuid][1].append(colour)
            # Fill candidates placed with this frame's cuboid.
            points = fill[uuid] @ rotation.T + center
            for view, surface in zip(views, person_depth):
                pixel, depth, inside = view.project(points)
                rows, cols = view.cells(pixel[inside])
                front = surface[rows, cols]
                nearest = view.nearest_around[rows, cols]
                offset = depth[inside] - front
                seen = (np.isfinite(front) & (offset >= BODY_DEPTH_M[0]) & (offset <= BODY_DEPTH_M[1])
                        & (nearest >= front - OCCLUSION_M))
                index = np.flatnonzero(inside)[seen]
                fill_seen[uuid][0].append(index)
                fill_seen[uuid][1].append(view.colours(pixel[index]))
    result, stats = {}, {}
    for uuid in sizes:
        points = np.concatenate(lidar[uuid][0]) if lidar[uuid][0] else np.empty((0, 3))
        colours = np.concatenate(lidar[uuid][1]) if lidar[uuid][1] else np.empty((0, 3))
        observed = colours[:, 0] >= 0
        owners = np.concatenate(fill_seen[uuid][0]) if fill_seen[uuid][0] else np.empty(0, dtype=np.int64)
        values = np.concatenate(fill_seen[uuid][1]).astype(np.float64) if fill_seen[uuid][1] else np.empty((0, 3))
        views = np.bincount(owners, minlength=len(fill[uuid]))
        fill_colour = median_by(owners, values, len(fill[uuid]))
        usable = np.flatnonzero(views >= min_views)
        reference = np.median(np.concatenate((colours[observed], fill_colour[usable])), axis=0) \
            if observed.any() or len(usable) else np.full(3, 128.0)
        # LiDAR returns no camera saw unoccluded keep their geometry and take the person's median colour.
        colours[~observed] = reference
        if len(points) > max_points:
            keep = generator.choice(len(points), max_points, replace=False)
            points, colours = points[keep], colours[keep]
        wanted = int(np.clip(target - len(points), 0, len(usable)))
        chosen = usable[np.argsort(-views[usable], kind="stable")[:wanted]] if wanted else usable[:0]
        result[uuid] = (np.concatenate((points, fill[uuid][chosen])).astype(np.float32),
                        np.concatenate((colours, fill_colour[chosen])).round().clip(0, 255).astype(np.uint8),
                        np.r_[np.zeros(len(points), np.uint8), np.ones(len(chosen), np.uint8)])
        stats[uuid] = {"lidar_points": int(len(points)), "lidar_coloured": int(observed.sum()),
                       "fill_points": int(len(chosen)), "fill_candidates_seen": int(len(usable)),
                       "size_m": [round(float(v), 3) for v in sizes[uuid]]}
    meta = {"schema_version": SCHEMA_VERSION, "scene_id": str(scene_id), "cameras": list(cameras),
            "target": target, "max_points": max_points, "min_views": min_views, "frame": "pandaset_cuboid",
            "pedestrians": stats}
    return result, meta


def save_seed(path, result, meta):
    uuids = sorted(result)
    owner = np.concatenate([np.full(len(result[u][0]), i, np.int32) for i, u in enumerate(uuids)]) \
        if uuids else np.empty(0, np.int32)
    stack = lambda i, empty: np.concatenate([result[u][i] for u in uuids]) if uuids else empty
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, uuids=np.array(uuids, dtype=str), owner=owner,
                        points=stack(0, np.empty((0, 3), np.float32)), colors=stack(1, np.empty((0, 3), np.uint8)),
                        source=stack(2, np.empty(0, np.uint8)), meta=np.array(json.dumps(meta)))
    temporary.replace(path)


def load_seed(path, scene_id):
    """uuid -> (box-frame points [N, 3], RGB 0-255 [N, 3], source [N] 0 LiDAR / 1 fill)."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta["schema_version"] != SCHEMA_VERSION or meta["scene_id"] != str(scene_id):
            raise ValueError("The deformable seed was built for another scene or format")
        owner = data["owner"]
        return {str(uuid): (data["points"][owner == i], data["colors"][owner == i], data["source"][owner == i])
                for i, uuid in enumerate(data["uuids"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    parser.add_argument("--target", type=int, default=2000, help="Points per pedestrian to reach with fill")
    parser.add_argument("--max-points", type=int, default=5000, help="Cap on LiDAR points (OmniRe instance_max_pts)")
    parser.add_argument("--min-views", type=int, default=2, help="Views that must see a fill point")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.target <= args.max_points or args.min_views < 1:
        parser.error("Use 0 < --target <= --max-points and --min-views >= 1")
    result, meta = build_seed(args.raw, args.scene, args.cameras, args.target, args.max_points,
                              min_views=args.min_views)
    save_seed(args.output, result, meta)
    for uuid, row in sorted(meta["pedestrians"].items()):
        print(f"{uuid[:8]}: {row['lidar_points']} LiDAR ({row['lidar_coloured']} có màu) + {row['fill_points']} điểm lấp")


if __name__ == "__main__":
    main()
