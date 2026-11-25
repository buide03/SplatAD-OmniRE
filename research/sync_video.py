"""Shared scene clock and H.264 encoding for the synchronized three-camera replay."""
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

FRONT_CAMERAS = ("front_left", "front", "front_right")


@dataclass(frozen=True)
class Timeline:
    start: float
    end: float
    fps: int

    @classmethod
    def from_tracks(cls, tracks, fps):
        if fps <= 0 or not tracks:
            raise ValueError("A positive FPS and at least one camera are required")
        for times in tracks:
            if len(times) < 2 or not all(math.isfinite(time) for time in times):
                raise ValueError("Each camera needs at least two finite timestamps")
            if any(a >= b for a, b in zip(times, times[1:])):
                raise ValueError("Source camera timestamps must increase")
        return cls(min(times[0] for times in tracks), max(times[-1] for times in tracks), fps)

    @property
    def frames(self):
        # Include the last observation even when the camera clock falls between output frames.
        return math.ceil((self.end - self.start) * self.fps - 1e-5) + 1

    def times(self):
        return (min(self.end, self.start + index / self.fps) for index in range(self.frames))

    def frame_times(self, frames, times):
        return {str(frame): round(time - self.start, 6) for frame, time in zip(frames, times, strict=True)}

    def video_metadata(self, path, width, height):
        return {"file": Path(path).name, "width": width, "height": height, "fps": self.fps,
                "frames": self.frames, "duration_s": self.frames / self.fps,
                "timeline_start_s": self.start}


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Install ffmpeg with libx264 or imageio-ffmpeg") from exc


def open_encoder(ffmpeg, path, width, height, fps, crf):
    return subprocess.Popen([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


def write_video(ffmpeg, destination, images, width, height, fps, crf):
    """Stream RGB frames to an atomic output and clean up a failed encode."""
    temporary = destination.with_suffix(".partial.mp4")
    encoder = open_encoder(ffmpeg, temporary, width, height, fps, crf)
    written = 0
    try:
        for image in images:
            encoder.stdin.write(image.tobytes())
            written += 1
        encoder.stdin.close()
        if encoder.wait() != 0 or not written:
            raise RuntimeError(f"ffmpeg could not encode {destination.name}")
        temporary.replace(destination)
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        if not encoder.stdin.closed:
            encoder.stdin.close()
        temporary.unlink(missing_ok=True)
    return written


def combine_front(ffmpeg, folder, videos, crf=16):
    """Stack independent files sharing the exact frame grid into one playback clock."""
    ordered = [videos[name] for name in FRONT_CAMERAS]
    first = ordered[0]
    keys = ("fps", "frames", "timeline_start_s")
    if any(any(video[key] != first[key] for key in keys) for video in ordered):
        raise ValueError("Three-camera replay requires the same timeline for every input")
    tile_width = min(1280, *(video["width"] for video in ordered)) // 2 * 2
    tile_height = min(720, *(video["height"] for video in ordered)) // 2 * 2
    if tile_width < 2 or tile_height < 2:
        raise ValueError("The replay resolution is too small for a triptych")
    # The 3840px width cap keeps the stitched video within common hardware decoder limits.
    filters = ";".join(
        f"[{i}:v]setpts=PTS-STARTPTS,scale={tile_width}:{tile_height}:"
        f"force_original_aspect_ratio=decrease:force_divisible_by=2,"
        f"pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
        for i in range(len(ordered))
    ) + ";[v0][v1][v2]hstack=inputs=3:shortest=1[fronts]"
    destination = folder / "front-triptych.mp4"
    temporary = destination.with_suffix(".partial.mp4")
    poster = destination.with_suffix(".jpg")
    poster_tmp = poster.with_suffix(".partial.jpg")
    inputs = [arg for video in ordered for arg in ("-i", str(folder / video["file"]))]
    try:
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs,
                        "-filter_complex_threads", "1", "-filter_complex", filters, "-map", "[fronts]",
                        "-r", str(first["fps"]), "-vsync", "cfr",
                        "-frames:v", str(first["frames"]), "-an", "-c:v", "libx264", "-preset", "fast",
                        "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)],
                       check=True)
        midpoint = min(first["duration_s"] / 2, 3.0)
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(midpoint),
                        "-i", str(temporary), "-frames:v", "1", "-vf", f"scale={min(1920, tile_width * 3)}:-2",
                        "-q:v", "2", str(poster_tmp)], check=True)
        poster_tmp.replace(poster)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
        poster_tmp.unlink(missing_ok=True)
    metadata = {
        "camera": "front_triptych", "cameras": list(FRONT_CAMERAS), "file": destination.name,
        "poster": poster.name,
        "fps": first["fps"], "frames": first["frames"], "timeline_start_s": first["timeline_start_s"],
        "duration_s": first["duration_s"], "width": 3 * tile_width, "height": tile_height,
        "frame_times_by_camera": {video["camera"]: video["frame_times"] for video in ordered},
    }
    if all(video.get("render_fps", 0) > 0 for video in ordered):
        metadata["render_fps"] = 1 / sum(1 / video["render_fps"] for video in ordered)
    return metadata
