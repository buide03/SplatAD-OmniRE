"""Render the complete scene at interpolated sensor poses and encode browser-ready MP4."""
import argparse
import math
import subprocess
from bisect import bisect_right
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch

from .evaluate import camera_indices, camera_sample, load_checkpoint, run_record, to_image
from .geometry import matrix_quat, quat_matrix, slerp
from .prepare import dump_json, read_json, sha256
from .source_preview import ffmpeg_executable


def camera_between(a, b, time, alpha):
    rot = slerp(matrix_quat(a.camera_to_worlds[..., :3]), matrix_quat(b.camera_to_worlds[..., :3]), alpha)
    origin = a.camera_to_worlds[..., 3:].lerp(b.camera_to_worlds[..., 3:], alpha)
    metadata = dict(a.metadata)
    metadata.pop("cam_idx", None)
    for key in ("time_to_center_pixel", "linear_velocities_local", "angular_velocities_local", "rolling_shutter_time"):
        if key in a.metadata and key in b.metadata:
            metadata[key] = a.metadata[key].lerp(b.metadata[key], alpha)
    return replace(a, camera_to_worlds=torch.cat((quat_matrix(rot), origin), dim=-1),
                   times=a.times.new_full(a.times.shape, time), metadata=metadata)


def open_encoder(ffmpeg, path, width, height, fps, crf):
    return subprocess.Popen([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


@torch.inference_mode()
def render_camera(pipeline, camera_name, output, fps, scale, crf, ffmpeg):
    manager, model = pipeline.datamanager, pipeline.model
    frame_indices = sorted((f, index) for (camera, f), index in camera_indices(manager).items() if camera == camera_name)
    if len(frame_indices) < 2:
        raise ValueError(f"At least two source frames are required for {camera_name}")
    views = [camera_sample(manager, index, with_image=False)[0] for _, index in frame_indices]
    times = [float(camera.times.item()) for camera in views]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError(f"Source camera timestamps are not increasing for {camera_name}")
    frame_count = math.floor((times[-1] - times[0]) * fps + 1e-5) + 1
    for view in views:
        if scale != 1:
            view.rescale_output_resolution(scale)
    width, height = int(views[0].width.item()), int(views[0].height.item())
    path = output / f"{camera_name}.mp4"
    temporary = output / f"{camera_name}.partial.mp4"
    elapsed_gpu = 0.0
    timeline = {}
    mark, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Warm lazy kernels and allocate render buffers before timing the sequence.
    model.get_camera_outputs(views[0])
    torch.cuda.synchronize()
    started = perf_counter()
    encoder = open_encoder(ffmpeg, temporary, width, height, fps, crf)
    try:
        for n in range(frame_count):
            time = times[0] + n / fps
            left = min(bisect_right(times, time) - 1, len(times) - 2)
            alpha = min(1.0, (time - times[left]) / (times[left + 1] - times[left]))
            mark.record()
            camera = camera_between(views[left], views[left + 1], time, views[left].times.new_tensor(alpha))
            rgb = model.get_camera_outputs(camera)["rgb"]
            end.record()
            image = to_image(rgb)
            end.synchronize()
            elapsed_gpu += mark.elapsed_time(end) / 1000
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg could not encode {camera_name}")
        temporary.replace(path)
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        temporary.unlink(missing_ok=True)
    for (frame, _), time in zip(frame_indices, times, strict=True):
        timeline[str(frame)] = round(time - times[0], 6)
    return {
        "camera": camera_name, "file": path.name, "fps": fps, "frames": frame_count,
        "width": width + width % 2, "height": height + height % 2, "duration_s": frame_count / fps,
        "render_fps": frame_count / max(elapsed_gpu, 1e-6),
        "export_fps": frame_count / max(perf_counter() - started, 1e-6),
        "source_fps": (len(times) - 1) / (times[-1] - times[0]), "frame_times": timeline,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--zero-smpl-velocity", action="store_true")
    args = parser.parse_args()
    ffmpeg = ffmpeg_executable()
    if not (1 <= args.fps <= 120 and 0 < args.scale <= 1 and 0 <= args.crf <= 51):
        parser.error("Use 1–120 FPS, a scale in (0,1], and a CRF in [0,51]")
    config, pipeline, checkpoint, step = load_checkpoint(args.config.resolve(), args.checkpoint)
    mode = "zero_smpl_velocity" if args.zero_smpl_velocity else "checkpoint"
    pipeline.model.human_motion = not args.zero_smpl_velocity
    manifest_path = args.manifest or config.pipeline.model.manifest_path
    if not manifest_path:
        parser.error("Provide the frozen manifest")
    if config.pipeline.model.manifest_digest and config.pipeline.model.manifest_digest != sha256(manifest_path):
        parser.error("The selected manifest differs from the one used in training")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "run.json"
    candidate = run_record(config, checkpoint, step, manifest_path, mode)
    record = read_json(metadata_path) if metadata_path.exists() else candidate
    if any(record[key] != candidate[key] for key in ("scene", "variant", "run_id", "step", "mode", "manifest_sha256")):
        parser.error("The output directory already belongs to another run")
    if not (args.output / "metrics.json").exists():
        record["metrics"] = None
    if not (args.output / "training.jsonl").exists():
        record["training"] = None
    cameras = args.cameras or list(config.pipeline.datamanager.dataparser.cameras)
    videos = {v["camera"]: v for v in record["videos"]}
    for camera in dict.fromkeys(cameras):
        videos[camera] = render_camera(pipeline, camera, args.output, args.fps, args.scale, args.crf, ffmpeg)
        record["videos"] = list(videos.values())
        dump_json(metadata_path, record)
        stats = videos[camera]
        print(f"{camera}: {stats['fps']} FPS playback, {stats['render_fps']:.1f} FPS GPU render, "
              f"{stats['export_fps']:.1f} FPS including encoding", flush=True)


if __name__ == "__main__":
    main()
