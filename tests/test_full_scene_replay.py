import json
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch

from research.pandaset_input import FRAMES_PER_SCENE, prepare_input, select_source
from research.replay import camera_center_offset
from research.sync_video import FRONT_CAMERAS, Timeline


def fixture_members(scene):
    result = {}
    stamps = json.dumps([frame / 10 for frame in range(FRAMES_PER_SCENE)]).encode()
    for name in FRONT_CAMERAS:
        camera = f"{scene}/camera/{name}_camera"
        result[f"{camera}/intrinsics.json"] = b"{}"
        result[f"{camera}/poses.json"] = b"[]"
        result[f"{camera}/timestamps.json"] = stamps
        result.update({f"{camera}/{frame:02d}.jpg": b"image" for frame in range(FRAMES_PER_SCENE)})
    result.update({f"{scene}/lidar/poses.json": b"[]", f"{scene}/lidar/timestamps.json": stamps})
    for folder in ("lidar", "annotations/cuboids"):
        result.update({f"{scene}/{folder}/{frame:02d}.pkl.gz": b"scan" for frame in range(FRAMES_PER_SCENE)})
    return result


def test_drive_reads_extracted_dataset_in_place_and_checks_last_frame(tmp_path):
    mydrive = tmp_path / "MyDrive"
    source = mydrive / "Research" / "data" / "pandaset"
    for relative, data in fixture_members("011").items():
        file = source / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
    staging = tmp_path / "staging"
    raw, scenes = prepare_input(None, [mydrive], staging, ["011"])
    assert raw == source and scenes == ["011"] and not staging.exists()
    single = select_source(source / "011", [], "all")
    assert single == (source, ["011"])
    (source / "011/camera/front_left_camera/79.jpg").unlink()
    with pytest.raises(ValueError, match="front_left_camera.*79.jpg"):
        prepare_input(source, [], staging, ["011"])


def test_drive_zip_stages_requested_scene_only(tmp_path):
    archive_file = tmp_path / "PandaSet.zip"
    with ZipFile(archive_file, "w") as archive:
        for scene in ("011", "028"):
            for relative, content in fixture_members(scene).items():
                archive.writestr(f"pandaset/{relative}", content)
        archive.writestr("pandaset/011/meta/../escape.txt", "forbidden")
    staging = tmp_path / "staging"
    root, requested = prepare_input(archive_file, [], staging, ["028"])
    assert requested == ["028"]
    assert (root / "028/camera/front_camera/79.jpg").read_bytes() == b"image"
    assert not (root / "011").exists() and not list(staging.rglob("escape.txt"))
    assert prepare_input(archive_file, [], staging, ["028"]) == (root, requested)
    assert select_source(archive_file, [], "all") == (archive_file, ["011", "028"])


def test_triptych_timeline_spans_first_and_last_capture_across_all_cameras():
    captures = [[0.002, 0.102, 0.202], [0.0, 0.1, 0.2], [0.008, 0.108, 0.208]]
    clock = Timeline.from_tracks(captures, 60)
    positions = list(clock.times())
    assert positions[0] == 0.0
    assert positions[-1] == 0.208
    assert len(positions) == clock.frames == 14
    assert positions == sorted(set(positions))
    assert clock.frame_times(range(3), captures[0]) == {"0": 0.002, "1": 0.102, "2": 0.202}
    assert clock.frame_times(range(3), captures[2])["2"] == 0.208
    with pytest.raises(ValueError, match="increase"):
        Timeline.from_tracks([[0.0, 0.0]], 60)


def test_capture_time_stays_shared_with_different_camera_shutter_offsets():
    model = SimpleNamespace(
        config=SimpleNamespace(compensate_rs_camera=True),
        camera_velocity_optimizer=SimpleNamespace(
            get_time_to_center_pixel_adjustment=lambda camera: camera.metadata["learned_offset"],
        ),
    )
    cameras = [
        SimpleNamespace(times=torch.tensor([[1.5]]),
                        metadata={"time_to_center_pixel": torch.tensor(nominal),
                                  "learned_offset": torch.tensor(learned)})
        for nominal, learned in [(-0.01, 0.002), (-0.02, 0.001), (0.004, -0.001)]
    ]
    center = camera_center_offset(model, cameras[1])
    effective = []
    for camera in cameras:
        corrected = camera.times + center - camera_center_offset(model, camera)
        effective.append(corrected + camera_center_offset(model, camera))
    assert torch.allclose(torch.stack(effective), torch.full((3, 1, 1), 1.481))
