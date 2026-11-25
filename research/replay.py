"""Render the complete scene at interpolated sensor poses and encode browser-ready MP4."""
import argparse
from bisect import bisect_right
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch

from .evaluate import camera_indices, camera_sample, load_checkpoint, run_record, to_image
from .geometry import matrix_quat, quat_matrix, slerp
from .prepare import dump_json, read_json, sha256
from .sync_video import FRONT_CAMERAS, Timeline, combine_front, ffmpeg_executable, write_video


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


def camera_track(manager, camera_name, indices):
    samples = sorted((frame, index) for (name, frame), index in indices.items() if name == camera_name)
    if len(samples) < 2:
        raise ValueError(f"At least two source frames are required for {camera_name}")
    views = [camera_sample(manager, index, with_image=False)[0] for _, index in samples]
    return [frame for frame, _ in samples], views, [float(camera.times.item()) for camera in views]


def camera_center_offset(model, camera):
    """Match SplatAD's RGB capture-time correction without changing camera geometry."""
    if not model.config.compensate_rs_camera or camera.metadata is None:
        return camera.times.new_zeros(())
    offset = camera.metadata.get("time_to_center_pixel", camera.times.new_zeros(()))
    return (offset + model.camera_velocity_optimizer.get_time_to_center_pixel_adjustment(camera)).reshape(())


@torch.inference_mode()
def render_camera(model, camera_name, track, output, timeline, scale, crf, ffmpeg, center_offset=None):
    frames, views, times = track
    for view in views:
        if scale != 1:
            view.rescale_output_resolution(scale)
    width, height = int(views[0].width.item()), int(views[0].height.item())
    path = output / f"{camera_name}.mp4"
    elapsed_gpu = 0.0
    mark, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Warm lazy kernels and allocate render buffers before timing the sequence.
    model.get_camera_outputs(views[0])
    torch.cuda.synchronize()
    started = perf_counter()

    def images():
        nonlocal elapsed_gpu
        for time in timeline.times():
            left = min(max(0, bisect_right(times, time) - 1), len(times) - 2)
            # At a camera's staggered endpoints, hold its pose while the scene clock continues.
            alpha = min(1.0, max(0.0, (time - times[left]) / (times[left + 1] - times[left])))
            mark.record()
            camera = camera_between(views[left], views[left + 1], time, views[left].times.new_tensor(alpha))
            if center_offset is not None:
                # The renderer adds its own offset to camera.times. Compensate the
                # input time so all views query the same scene at the front-camera center.
                camera.times = camera.times + center_offset - camera_center_offset(model, camera)
            rgb = model.get_camera_outputs(camera)["rgb"]
            end.record()
            image = to_image(rgb)
            end.synchronize()
            elapsed_gpu += mark.elapsed_time(end) / 1000
            yield image

    frame_count = write_video(ffmpeg, path, images(), width, height, timeline.fps, crf)
    if frame_count != timeline.frames:
        raise RuntimeError(f"Incomplete replay for {camera_name}")
    return {
        **timeline.video_metadata(path, width + width % 2, height + height % 2),
        "camera": camera_name,
        "render_fps": frame_count / max(elapsed_gpu, 1e-6),
        "export_fps": frame_count / max(perf_counter() - started, 1e-6),
        "source_fps": (len(times) - 1) / (times[-1] - times[0]),
        "frame_times": timeline.frame_times(frames, times),
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
    candidate = run_record(config, checkpoint, step, manifest_path, read_json(manifest_path), mode)
    record = read_json(metadata_path) if metadata_path.exists() else candidate
    if any(record[key] != candidate[key] for key in ("scene", "variant", "run_id", "step", "mode", "manifest_sha256")):
        parser.error("The output directory already belongs to another run")
    record["pose_summary"] = candidate["pose_summary"]
    if not (args.output / "metrics.json").exists():
        record["metrics"] = None
    if not (args.output / "training.jsonl").exists():
        record["training"] = None
    cameras = list(dict.fromkeys(args.cameras or list(config.pipeline.datamanager.dataparser.cameras)))
    indices = camera_indices(pipeline.datamanager)
    tracks = {camera: camera_track(pipeline.datamanager, camera, indices) for camera in cameras}
    synchronized = all(name in tracks for name in FRONT_CAMERAS)
    front_clock = Timeline.from_tracks([tracks[name][2] for name in FRONT_CAMERAS], args.fps) if synchronized else None
    with torch.inference_mode():
        center_offset = camera_center_offset(pipeline.model, tracks["front"][1][0]) if synchronized else None
    if any(name in FRONT_CAMERAS for name in cameras) and record.pop("front_triptych", None):
        dump_json(metadata_path, record)
    videos = {v["camera"]: v for v in record["videos"]}
    for camera in cameras:
        front_view = synchronized and camera in FRONT_CAMERAS
        clock = front_clock if front_view else Timeline.from_tracks([tracks[camera][2]], args.fps)
        videos[camera] = render_camera(pipeline.model, camera, tracks[camera], args.output, clock,
                                       args.scale, args.crf, ffmpeg, center_offset if front_view else None)
        record["videos"] = list(videos.values())
        dump_json(metadata_path, record)
        stats = videos[camera]
        print(f"{camera}: {stats['fps']} FPS playback, {stats['render_fps']:.1f} FPS GPU render, "
              f"{stats['export_fps']:.1f} FPS including encoding", flush=True)
    if synchronized:
        record["front_triptych"] = combine_front(ffmpeg, args.output, videos, args.crf)
        dump_json(metadata_path, record)
        print(f"Three synchronized views: {args.output / record['front_triptych']['file']}", flush=True)


if __name__ == "__main__":
    main()
