"""Preview the original PandaSet camera sequence alongside frozen observations."""
import argparse
from bisect import bisect_left
from pathlib import Path

from PIL import Image

from .prepare import camera_at, dump_json, front_pose_summary, read_json, sha256, validate_tracks
from .sync_video import FRONT_CAMERAS, Timeline, combine_front, ffmpeg_executable, write_video


def initial_people(tracks_path, body_path, manifest, time_offset):
    import torch
    from .geometry import BodyModel
    from .smpl_nodes import SMPLTrack

    document = validate_tracks(read_json(tracks_path), manifest["scene_id"])
    if sha256(body_path) != document["body_sha256"]:
        raise ValueError("The preview mesh does not match the pose JSON")
    body = BodyModel(body_path)
    want = {row["uuid"] for row in manifest["observations"] if row["probe"] and row["smpl_valid"]}
    people = {track["uuid"]: SMPLTrack(track, body, torch.eye(4), time_offset, 13, 0.1, False)
              for track in document["tracks"] if track["uuid"] in want}
    return people, body.version


def mesh_preview(image, person, raw_scene, camera, frame, time_offset):
    import numpy as np
    import torch
    from .evaluate import overlay, project

    rotation, origin, k, time = camera_at(raw_scene, camera, frame)
    view = torch.tensor(np.column_stack([rotation.T, -rotation.T @ origin]), dtype=torch.float32)
    intrinsics = torch.tensor([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    with torch.no_grad():
        points, visible = person.mesh_at(torch.tensor(time - 0.01 - time_offset))
        if not bool(visible):
            return None
        return Image.fromarray(overlay(np.asarray(image), project(points, view, intrinsics), True,
                                       person.faces, (253, 197, 87)))


def source_track(scene, name):
    source = scene / "camera" / f"{name}_camera"
    frames = sorted(source.glob("*.jpg"))
    if len(frames) < 2 or [int(path.stem) for path in frames] != list(range(len(frames))):
        raise ValueError(f"Expected a contiguous sequence starting at 00.jpg in {source}")
    times = read_json(source / "timestamps.json")
    if len(times) != len(frames):
        raise ValueError(f"Frame and timestamp counts differ in {source}")
    return frames, times


def render_source(ffmpeg, name, track, output, shared_clock):
    frames, times = track
    fps = 10
    Timeline.from_tracks([times], fps)
    clock = shared_clock or Timeline(0.0, (len(frames) - 1) / fps, fps)
    destination = output / f"{name}.mp4"
    height = 820 if name == "back" else 1080
    with Image.open(frames[0]) as image:
        width = image.width
        height = min(height, image.height)

    def images():
        last = None
        image = None
        for frame, time in enumerate(clock.times()):
            if shared_clock:
                right = min(bisect_left(times, time), len(times) - 1)
                left = max(0, right - 1)
                frame = left if abs(times[left] - time) <= abs(times[right] - time) else right
            if frame != last:
                with Image.open(frames[frame]) as original:
                    image = original.convert("RGB").crop((0, 0, width, height))
                last = frame
            yield image

    count = write_video(ffmpeg, destination, images(), width, height, fps, 18)
    if count != clock.frames:
        raise RuntimeError(f"Incomplete preview for {name}")
    poster = output / f"{name}-poster.jpg"
    with Image.open(frames[min(30, len(frames) - 1)]) as image:
        image = image.crop((0, 0, width, height))
        image.thumbnail((1280, 720))
        image.save(poster, quality=90)
    captured = times if shared_clock else clock.times()
    return {
        **clock.video_metadata(destination, width + width % 2, height + height % 2),
        "camera": name, "poster": poster.name,
        "source_fps": (len(frames) - 1) / (times[-1] - times[0]),
        "frame_times": clock.frame_times(range(len(frames)), captured),
    }


def observation_panels(manifest, camera_tracks, people, raw_scene, output, offset):
    observations = []
    for index, row in enumerate(manifest["observations"]):
        name = row["camera"]
        if name not in camera_tracks or not row["usable"]:
            continue
        panels = {}
        if row["probe"]:
            panel = f"source-crops/{name}-{index:04d}.jpg"
            (output / panel).parent.mkdir(exist_ok=True)
            frames, _ = camera_tracks[name]
            with Image.open(frames[row["frame"]]) as image:
                image.crop(row["bbox_xyxy"]).save(output / panel, quality=92)
                if row["uuid"] in people:
                    overlay = mesh_preview(image, people[row["uuid"]], raw_scene, name, row["frame"], offset)
                    if overlay is not None:
                        overlay_panel = f"source-crops/{name}-{index:04d}-mesh.jpg"
                        overlay.crop(row["bbox_xyxy"]).save(output / overlay_panel, quality=92)
                        panels["initial"] = overlay_panel
            panels["gt"] = panel
        observations.append({**row, "panels": panels})
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene")
    parser.add_argument("--camera", default="front_right")
    parser.add_argument("--front-triptych", action="store_true")
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--body", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest) if args.manifest else None
    scene = args.scene or (manifest["scene_id"] if manifest else None)
    if not scene or (manifest and scene != manifest["scene_id"]):
        parser.error("Provide --scene or a matching --manifest")
    if bool(args.tracks) != bool(args.body) or (args.tracks and not manifest):
        parser.error("SMPL overlays require --tracks, --body and --manifest")
    raw_scene = args.raw / scene
    cameras = FRONT_CAMERAS if args.front_triptych else (args.camera,)
    tracks = {name: source_track(raw_scene, name) for name in cameras}
    clock = Timeline.from_tracks([entry[1] for entry in tracks.values()], 10) if args.front_triptych else None
    ffmpeg = ffmpeg_executable()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    videos = {name: render_source(ffmpeg, name, track, output, clock) for name, track in tracks.items()}
    offset = min(entry[1][0] for entry in tracks.values())
    people, version = initial_people(args.tracks, args.body, manifest, offset) if args.tracks else ({}, None)
    if manifest:
        observations = observation_panels(manifest, tracks, people, raw_scene, output, offset)
        dump_json(output / "metrics.json", {"scene": scene, "observations": observations,
                                            "scene_metrics": [], "coverage": manifest["coverage"]})
    record = {
        "scene": scene, "variant": "GT", "mode": "source", "step": None, "run_id": "source",
        "manifest_sha256": sha256(args.manifest) if manifest else None,
        "pose_summary": front_pose_summary(manifest) if manifest else None,
        "metrics": "metrics.json" if manifest else None, "training": None,
        "smpl_version": version, "videos": list(videos.values()),
    }
    if clock:
        record["front_triptych"] = combine_front(ffmpeg, output, videos, crf=18)
    dump_json(output / "run.json", record)
    print(f"{scene}: {len(videos)} camera previews in {output}")


if __name__ == "__main__":
    main()
