"""CPU checks for following every SMPL mesh during training (R2/R3) and per checkpoint (OmniRe)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research import pose_monitor
from research.omnire_baseline import omnire_drift, smpl_people
from research.pose_monitor import PoseMonitor, pose_drift, track_observations

IDENTITY = [1., 0., 0., 0.]


def observation(uuid, camera, frame, height, **extra):
    return {"uuid": uuid, "camera": camera, "frame": frame, "bbox_height_px": height, "usable": True,
            "smpl_valid": True, "probe": False, "bbox_xyxy": [0, 0, 4, height], **extra}


def test_one_clear_observation_per_person_from_any_camera():
    rows = [
        observation("a", "front", 3, 120), observation("a", "back", 4, 300, limbs_visible=False),
        observation("a", "left", 5, 200), observation("b", "right", 2, 90),
        observation("b", "front", 9, 400, usable=False), observation("c", "front", 1, 500, smpl_valid=False),
        observation("taken", "front", 1, 600),
    ]
    cameras = {(row["camera"], row["frame"]): i for i, row in enumerate(rows)}
    chosen = track_observations({"observations": rows}, {"a", "b", "c", "taken"}, cameras, {"taken"})
    assert [(row["uuid"], row["camera"], row["frame"]) for row in chosen] == [("a", "left", 5), ("b", "right", 2)]


def test_pose_drift_reports_centimeters_and_degrees():
    angle = torch.tensor(np.deg2rad(10.0) / 2)
    turned = torch.stack((angle.cos(), torch.tensor(0.), torch.tensor(0.), angle.sin()))
    current = {"translation": torch.tensor([[0.03, 0., 0.], [0., 0., 0.]]),
               "root": torch.stack((turned, torch.tensor(IDENTITY)))}
    initial = {"translation": torch.zeros(2, 3), "root": torch.tensor([IDENTITY, IDENTITY])}
    drift = pose_drift(current, initial)
    assert drift["translation"]["unit"] == "cm" and drift["translation"]["max"] == pytest.approx(3)
    assert drift["root"]["unit"] == "deg" and drift["root"]["max"] == pytest.approx(10, abs=0.05)
    assert drift["root"]["mean"] == pytest.approx(5, abs=0.05)


class Track(torch.nn.Module):
    def __init__(self, uuid, shift):
        super().__init__()
        self.uuid = uuid
        self.translation = torch.nn.Parameter(torch.tensor([[shift, 0., 0.]]))
        self.root = torch.nn.Parameter(torch.tensor([IDENTITY]))
        self.body = torch.nn.Parameter(torch.tensor([[IDENTITY] * 23]))
        self.initial_translation = torch.zeros(1, 3)
        self.initial_root = torch.tensor([IDENTITY])
        self.initial_body = torch.tensor([[IDENTITY] * 23])


def test_monitor_follows_every_smpl_pedestrian_and_probes_only_reviewed_ones(tmp_path, monkeypatch):
    reviewed = observation("a", "front", 4, 100, pose_monitor=True)
    rows = [reviewed, observation("b", "back", 6, 80), observation("b", "left", 7, 150)]
    people = [Track("a", 0.0), Track("b", 0.05)]
    model = SimpleNamespace(person_index={"a": 0, "b": 1}, people=people,
                            config=SimpleNamespace(variant="R3", use_camopt_in_eval=False),
                            camera_velocity_optimizer=SimpleNamespace(config=SimpleNamespace(enabled=False)),
                            training=True, train=lambda mode=True: None, eval=lambda: None)
    files = [tmp_path / "front_camera" / "04.jpg", tmp_path / "back_camera" / "06.jpg", tmp_path / "left_camera" / "07.jpg"]
    manager = SimpleNamespace(train_dataset=SimpleNamespace(image_filenames=files),
                              cached_train=[{} for _ in files])
    manifest = {"scene_id": "030", "selected_uuids": ["a", "b"], "observations": rows}
    monitor = PoseMonitor(SimpleNamespace(model=model, datamanager=manager), manifest, tmp_path / "out", "run", 0, "d")
    assert {key: row["probe_update"] for key, row in monitor.observations.items()} == {
        "front/4/a": True, "left/7/b": False}
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(pose_monitor, "camera_sample", lambda manager, index: (None, None))
    monkeypatch.setattr(pose_monitor, "measure_rgb", lambda *args, **kwargs: (
        {"rgb_mae": 0.1, "silhouette_iou": None}, {"reference": image, "initial": image, "current": image}))
    probed = []
    monkeypatch.setattr(pose_monitor, "pose_only_rgb_step", lambda model, person, *args: (
        probed.append(person.uuid) or {"status": "measured"}, {}))
    monitor.capture(250, [3e-4, 2e-4, 2e-4])
    step = monitor.steps[-1]
    assert probed == ["a"]
    assert step["results"]["left/7/b"]["pose_only_rgb_probe"]["status"] == "tracked"
    assert (tmp_path / "out" / step["results"]["left/7/b"]["panels"]["current"]).is_file()
    assert step["tracks"]["b"]["translation"]["max"] == pytest.approx(5)
    assert step["tracks"]["a"]["root"]["max"] == pytest.approx(0)


def test_omnire_instances_follow_drivestudio_order_and_moving_filter():
    still = {"frame_valid": torch.tensor([True, True]), "smpl_trans": torch.zeros(2, 3)}
    walking = {"frame_valid": torch.tensor([True, True]), "smpl_trans": torch.tensor([[0., 0., 0.], [1., 0., 0.]])}
    unseen = {"frame_valid": torch.tensor([False, False]), "smpl_trans": torch.zeros(2, 3)}
    source = SimpleNamespace(instances_true_id=torch.tensor([4, 2, 7, 9]),
                             smpl_human_all={4: walking, 2: still, 7: unseen, 3: walking})
    dataset = SimpleNamespace(instance_num=4, pixel_source=source)
    info = {str(i): {"id": f"uuid-{i}"} for i in range(10)}
    moving = SimpleNamespace(model=SimpleNamespace(SMPLNodes={"init": {"only_moving": True, "traj_length_thres": 0.5}}))
    assert smpl_people(dataset, moving, info) == [(4, "uuid-4")]
    every = SimpleNamespace(model=SimpleNamespace(SMPLNodes={"init": {"only_moving": False}}))
    assert smpl_people(dataset, every, info) == [(4, "uuid-4"), (2, "uuid-2")]


def test_omnire_drift_uses_only_frames_with_pose():
    node = SimpleNamespace(
        instances_fv=torch.tensor([[True], [False]]),
        instances_trans=torch.tensor([[[0.02, 0., 0.]], [[5., 0., 0.]]]),
        instances_quats=torch.tensor([IDENTITY, IDENTITY]).reshape(2, 1, 1, 4),
        smpl_qauts=torch.tensor([IDENTITY] * 46).reshape(2, 1, 23, 4),
    )
    initial = {"instances_trans": torch.zeros(2, 1, 3), "instances_quats": node.instances_quats.clone(),
               "smpl_qauts": node.smpl_qauts.clone()}
    drift = omnire_drift(node, initial, 0)
    assert drift["translation"]["max"] == pytest.approx(2)
    assert drift["body"]["max"] == pytest.approx(0)
