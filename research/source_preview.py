"""Preview the original PandaSet camera sequence alongside frozen observations."""
import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from .prepare import camera_at, dump_json, read_json, sha256, validate_tracks


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Install ffmpeg with libx264 or imageio-ffmpeg") from exc


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera", default="front_right")
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--body", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    scene = manifest["scene_id"]
    name = args.camera
    source = args.raw / scene / "camera" / f"{name}_camera"
    frames = sorted(source.glob("*.jpg"))
    if len(frames) < 2 or [int(path.stem) for path in frames] != list(range(len(frames))):
        parser.error("Expected a contiguous sequence starting at 00.jpg")
    times = read_json(source / "timestamps.json")
    fps = (len(frames) - 1) / (times[len(frames) - 1] - times[0])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{name}.mp4"
    temporary = output / f"{name}.partial.mp4"
    height = 820 if name == "back" else 1080
    with Image.open(frames[0]) as image:
        width = image.width
        height = min(height, image.height)
    steps_per_second = 10
    try:
        subprocess.run([
            ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(steps_per_second), "-i", str(source / "%02d.jpg"),
            "-frames:v", str(len(frames)), "-vf", f"crop={width}:{height}:0:0",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
        ], check=True)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    if bool(args.tracks) != bool(args.body):
        parser.error("SMPL overlays require --tracks and --body together")
    people, smpl_version = initial_people(args.tracks, args.body, manifest, times[0]) if args.tracks else ({}, None)
    observations = []
    for index, row in enumerate(manifest["observations"]):
        if row["camera"] != name or not row["usable"]:
            continue
        panels = {}
        if row["probe"]:
            panel = f"source-crops/{name}-{index:04d}.jpg"
            (output / panel).parent.mkdir(exist_ok=True)
            with Image.open(frames[row["frame"]]) as image:
                image.crop(row["bbox_xyxy"]).save(output / panel, quality=92)
                if row["uuid"] in people:
                    overlay = mesh_preview(image, people[row["uuid"]], args.raw / scene, name, row["frame"], times[0])
                    if overlay is not None:
                        overlay_panel = f"source-crops/{name}-{index:04d}-mesh.jpg"
                        overlay.crop(row["bbox_xyxy"]).save(output / overlay_panel, quality=92)
                        panels["initial"] = overlay_panel
            panels["gt"] = panel
        observations.append({**row, "panels": panels})
    poster = output / f"{name}-poster.jpg"
    with Image.open(frames[min(30, len(frames) - 1)]) as image:
        image = image.crop((0, 0, width, height))
        image.thumbnail((1280, 720))
        image.save(poster, quality=90)
    dump_json(output / "metrics.json", {"scene": scene, "observations": observations,
                                        "scene_metrics": [], "coverage": manifest["coverage"]})
    dump_json(output / "run.json", {
        "scene": scene, "variant": "GT", "mode": "source", "step": None, "run_id": "source",
        "manifest_sha256": sha256(args.manifest), "metrics": "metrics.json", "training": None,
        "smpl_version": smpl_version,
        "videos": [{"camera": name, "file": destination.name, "poster": poster.name,
                    "fps": steps_per_second, "source_fps": fps, "frames": len(frames), "width": width,
                    "height": height, "duration_s": len(frames) / steps_per_second,
                    "frame_times": {str(i): i / steps_per_second for i in range(len(frames))}}],
    })
    print(f"{destination} ({len(frames)} source frames)")


if __name__ == "__main__":
    main()
