"""Resolve PandaSet on mounted Drive and stage requested scenes from a ZIP when necessary."""
import hashlib
import json
import re
import shutil
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .download import scene_files
from .sync_video import FRONT_CAMERAS

SCENE_ID = re.compile(r"[0-9]{3}")
FRAMES_PER_SCENE = 80


def scene_dirs(folder):
    return sorted(path.name for path in folder.iterdir() if SCENE_ID.fullmatch(path.name)
                  and (path / "camera" / "front_camera").is_dir()
                  and (path / "lidar").is_dir() and (path / "annotations" / "cuboids").is_dir())


def archive_scenes(archive):
    return sorted({part for entry in archive.infolist()
                   for part in re.findall(r"(?:^|/)([0-9]{3})/camera/front_camera/intrinsics\.json$", entry.filename)})


def find_candidates(mounts, depth=5):
    names = ("pandaset", "panda set", "panda-set", "panda_set")
    seen = set()
    pending = [(Path(folder), 0) for folder in mounts]
    while pending:
        path, level = pending.pop(0)
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            scenes = scene_dirs(path)
            if scenes:
                yield path, scenes
                continue
            for child in path.iterdir():
                if child.name.startswith("."):
                    continue
                if child.suffix.lower() == ".zip" and any(name in child.stem.lower() for name in names):
                    try:
                        with ZipFile(child) as archive:
                            supported = archive_scenes(archive)
                    except (OSError, BadZipFile):
                        continue
                    if supported:
                        yield child, supported
                elif level < depth and child.is_dir():
                    pending.append((child, level + 1))
        except OSError:
            continue


def select_source(path, mounts, scenes):
    if path:
        root = Path(path).expanduser()
        if root.is_dir():
            if SCENE_ID.fullmatch(root.name) and (root / "camera").is_dir():
                root = root.parent
            available = scene_dirs(root)
            if not available:
                # Also accept an extracted wrapper such as dataset/pandaset.
                candidates = list(find_candidates([root], depth=3))
            else:
                candidates = [(root, available)]
        elif root.is_file() and root.suffix.lower() == ".zip":
            with ZipFile(root) as archive:
                candidates = [(root, archive_scenes(archive))]
        else:
            raise ValueError(f"PANDASET_PATH is not a folder or a ZIP on mounted Drive: {root}")
    else:
        candidates = list(find_candidates(mounts))

    wanted = set(scenes) if scenes != "all" else set()
    candidates = [(root, available) for root, available in candidates if available and wanted <= set(available)]
    if not candidates:
        raise ValueError("PandaSet was not found for the requested scenes. Set PANDASET_PATH to the raw "
                         "folder containing 011/camera or to the PandaSet ZIP on mounted Drive.")
    candidates.sort(key=lambda item: (len(item[1]), item[0].is_dir()), reverse=True)
    root, available = candidates[0]
    tied = [candidate for candidate, found in candidates if len(found) == len(available)
            and candidate.is_dir() == root.is_dir()]
    if len(tied) > 1:
        paths = "\n".join(str(candidate) for candidate in tied)
        raise ValueError(f"Multiple PandaSet copies match. Set PANDASET_PATH to one of:\n{paths}")
    return root, available if scenes == "all" else list(dict.fromkeys(scenes))


def stage_archive(source, staging, scenes):
    stat = source.stat()
    fingerprint = hashlib.sha256(f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:12]
    root = Path(staging) / fingerprint
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    with ZipFile(source) as archive:
        selected = [(entry, relative) for entry, relative in scene_files(archive, set(scenes))
                    if "\\" not in entry.filename and ":" not in entry.filename]
        bytes_left = 0
        for entry, relative in selected:
            target = (root / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Invalid dataset ZIP member: {relative}")
            if not target.is_file() or target.stat().st_size != entry.file_size:
                bytes_left += entry.file_size
        if bytes_left > shutil.disk_usage(root).free:
            raise ValueError(f"Selected scenes require {bytes_left / 2**30:.1f} GiB of local staging. "
                             "Select fewer scenes or point PANDASET_PATH to the extracted folder on Drive.")
        for entry, relative in selected:
            target = (root / relative).resolve()
            if target.is_file() and target.stat().st_size == entry.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            with archive.open(entry) as inp, temporary.open("wb") as out:
                shutil.copyfileobj(inp, out)
            temporary.replace(target)
    return root


def check_scene(root, scene, cameras):
    folder = root / scene
    missing = []

    def files(directory, expected):
        available = {item.name for item in directory.iterdir()} if directory.is_dir() else set()
        missing.extend(str(directory.relative_to(folder) / name) for name in sorted(set(expected) - available))

    for name in set(cameras) | {"front"}:
        camera = folder / "camera" / f"{name}_camera"
        files(camera, ["intrinsics.json", "poses.json", "timestamps.json",
                       *(f"{frame:02d}.jpg" for frame in range(FRAMES_PER_SCENE))])
    files(folder / "lidar", ["poses.json", "timestamps.json",
                            *(f"{frame:02d}.pkl.gz" for frame in range(FRAMES_PER_SCENE))])
    files(folder / "annotations" / "cuboids", [f"{frame:02d}.pkl.gz" for frame in range(FRAMES_PER_SCENE)])
    if missing:
        raise ValueError(f"Incomplete PandaSet scene {scene} at {folder}. Missing {len(missing)} files, "
                         f"including {', '.join(missing[:5])}.")
    for name in set(cameras) | {"front"}:
        stamps = folder / "camera" / f"{name}_camera" / "timestamps.json"
        if len(json.loads(stamps.read_text(encoding="utf-8"))) != FRAMES_PER_SCENE:
            raise ValueError(f"{scene}/{name}: expected {FRAMES_PER_SCENE} timestamps in {stamps}")


def prepare_input(path, mounts, staging, scenes, cameras=FRONT_CAMERAS):
    if scenes != "all" and (isinstance(scenes, str) or not scenes
                            or any(not SCENE_ID.fullmatch(scene) for scene in scenes)):
        raise ValueError('SCENES must be a list of three-digit IDs or "all"')
    source, selected = select_source(path, mounts, scenes)
    root = source.resolve() if source.is_dir() else stage_archive(source, staging, selected)
    for scene in selected:
        check_scene(root, scene, cameras)
    return root, selected
