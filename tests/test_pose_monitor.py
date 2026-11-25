from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from research.pose_monitor import PoseMonitor, measure_rgb, polygon_mask, pose_only_rgb_step, select_observations
from research.prepare import front_pose_summary


class Person(nn.Module):
    def __init__(self):
        super().__init__()
        self.translation = nn.Parameter(torch.zeros((2, 3)))
        self.root = nn.Parameter(torch.tensor([[1., 0., 0., 0.]] * 2))
        self.body = nn.Parameter(torch.tensor([[[1., 0., 0., 0.]]] * 2))
        self.register_buffer("times", torch.tensor([0., 1.]))
        self.register_buffer("frame_ids", torch.tensor([4, 5]))

    def _span(self, _):
        return torch.tensor(0), torch.tensor(0), torch.tensor(0.), torch.tensor(True)


class Scene(nn.Module):
    def __init__(self):
        super().__init__()
        self.person = Person()
        self.scene_parameter = nn.Parameter(torch.tensor(0.5))
        self.last_actor_time = torch.tensor(1.0)
        self.last_rgb_index = 91
        self.last_sensor = "lidar"
        self.last_size = (8, 9)
        self.info = {"old": 1}

    def get_camera_outputs(self, camera):
        self.last_actor_time = torch.tensor(0.0)
        self.last_size = (9, 8)
        self.info = {}
        rgb = self.person.translation[0, 0] + self.scene_parameter + torch.rand(()) * 0.01
        return {"rgb": rgb}

    def get_loss_dict(self, outputs, batch):
        self.last_sensor, self.last_rgb_index = "rgb", 4
        loss = (outputs["rgb"] - batch["target"]).square()
        loss = loss + (self.person.root[0, 1] - 0.4).square() + (self.person.body[0, 0, 1] - 0.2).square()
        return {"main_loss": loss, "scene_regularizer": self.scene_parameter.square()}


def test_pose_step_has_gradient_but_restores_live_training_state():
    model = Scene().eval()
    actual = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.person.translation.grad = torch.full_like(model.person.translation, 7)
    rng = torch.get_rng_state().clone()

    def measure():
        assert not model.training
        assert model.scene_parameter.item() == 0.5
        assert not model.person.translation[1].any()
        return {"rgb_mae": float((model.person.translation[0, 0] - 1.).abs())}, {"current": "panel"}

    record, panels = pose_only_rgb_step(
        model, model.person, lambda: (None, {"target": 1.5}), [0.1] * 3, measure
    )
    assert record["status"] == "measured"
    assert record["active_frames"] == [4]
    assert record["loss_after"] < record["loss_before"]
    assert record["after"]["rgb_mae"] < 1
    assert panels == {"current": "panel"}
    assert all(g["state"] == "finite" and g["delta"]["max"] > 0 for g in record["groups"].values())
    assert all(torch.equal(p, actual[name]) for name, p in model.named_parameters())
    assert torch.equal(model.person.translation.grad, torch.full_like(model.person.translation, 7))
    assert torch.equal(torch.get_rng_state(), rng) and not model.training
    assert model.last_rgb_index == 91 and model.last_sensor == "lidar" and model.info == {"old": 1}


def test_pose_step_restores_pose_when_after_measurement_fails():
    model = Scene().train()
    before = model.person.translation.detach().clone()

    def fail():
        raise RuntimeError("measurement failed")

    with pytest.raises(RuntimeError, match="measurement failed"):
        pose_only_rgb_step(model, model.person, lambda: (None, {"target": 1.5}), [0.1] * 3, fail)
    assert torch.equal(model.person.translation, before) and model.training


def test_frozen_pose_skips_gradient_probe():
    model = Scene()
    model.person.requires_grad_(False)
    record, panels = pose_only_rgb_step(model, model.person, None, [], None)
    assert record == {"status": "frozen"} and not panels


def test_rgb_mae_uses_fixed_gt_polygon_and_mesh_iou():
    polygon = [[22, 22], [32, 22], [32, 32], [22, 32]]
    image = torch.zeros((64, 64, 3))
    target = polygon_mask((64, 64), polygon)
    render = torch.ones_like(image)
    render[target] = 0.125
    world = torch.eye(4)[:3]
    model = SimpleNamespace(
        get_camera_outputs=lambda camera: {"rgb": render, "background": torch.zeros((3,)),
                                           "world_to_camera": world, "capture_time": torch.tensor(0.)},
        get_gt_img=lambda gt: gt, composite_with_background=lambda gt, bg: gt,
    )
    camera = SimpleNamespace(get_intrinsics_matrices=lambda: torch.eye(3)[None])

    def mesh_at(time, initial=False):
        dx = 10 if initial else 0
        vertices = torch.tensor([[x + dx, y, 1.0] for x, y in polygon])
        return vertices, torch.tensor(True)

    person = SimpleNamespace(faces=torch.tensor([[0, 1, 2], [0, 2, 3]]), mesh_at=mesh_at)
    frame = {"bbox_xyxy": [19, 20, 35, 34], "silhouette_xy": polygon,
             "keypoints": {"corner": {"xy": [22, 22], "vertex_id": 0}}}
    metrics, panels = measure_rgb(model, camera, image, person, frame)
    assert metrics["roi_source"] == "manual_silhouette"
    assert metrics["rgb_mae"] == pytest.approx(0.125)
    assert metrics["roi_pixels"] == int(np.count_nonzero(target))
    assert metrics["silhouette_iou"] == 1.0 > metrics["silhouette_initial_iou"]
    assert metrics["reprojection_mae_px"] == 0.0 < metrics["reprojection_initial_mae_px"]
    assert set(panels) == {"reference", "initial", "current", "render", "error"}

    frame.pop("silhouette_xy")
    metrics, _ = measure_rgb(model, camera, image, person, frame)
    assert metrics["roi_source"] == "fixed_bbox_crop"
    assert metrics["silhouette_iou"] is None and metrics["rgb_mae"] > 0.125


def test_monitor_selects_reviewed_observation_and_logs_exact_rgb_frame(tmp_path):
    base = {"uuid": "actor", "usable": True, "smpl_valid": True, "camera": "front",
            "frame": 4, "bbox_height_px": 100, "probe": True}
    chosen = {**base, "frame": 5, "bbox_height_px": 50, "pose_monitor": True}
    manifest = {"scene_id": "011", "observations": [base, chosen]}
    model = SimpleNamespace(person_index={"actor": 0}, config=SimpleNamespace(variant="R2"))
    manager = SimpleNamespace(train_dataset=SimpleNamespace(image_filenames=[
        tmp_path / "front_camera" / "04.jpg", tmp_path / "front_camera" / "05.jpg",
    ]))
    cameras = {("front", 4): 0, ("front", 5): 1}
    assert select_observations(manifest, model, cameras) == [chosen]
    monitor = PoseMonitor(SimpleNamespace(model=model, datamanager=manager), manifest, tmp_path / "out",
                          "example", 0, "digest")
    log = {"step": 10, "sensor": "rgb", "image_index": 0, "scene_loss": 0.4,
           "pose": {"actor": {"translation": {"step_delta": {"max": 1}}}}}
    monitor.observe_training_step(log)
    monitor.observe_training_step({**log, "image_index": 1, "sensor": "lidar"})
    assert not monitor.actual
    monitor.observe_training_step({**log, "image_index": 1})
    assert monitor.actual["front/5/actor"]["step"] == 11


def test_monitor_can_inspect_an_explicit_occluded_observation_without_scoring_it_as_visible():
    base = {"uuid": "actor", "camera": "front", "smpl_valid": True, "bbox_height_px": 70,
            "usable": False, "limbs_visible": False, "probe": False}
    selected = {**base, "frame": 16, "pose_monitor": True}
    unmarked = {**base, "frame": 19, "bbox_height_px": 95}
    unreviewed = {**base, "frame": 22, "limbs_visible": None, "pose_monitor": True}
    rows = [selected, unmarked, unreviewed]
    cameras = {("front", row["frame"]): index for index, row in enumerate(rows)}
    model = SimpleNamespace(person_index={"actor": 0})
    assert select_observations({"observations": rows}, model, cameras) == [selected]


def test_pose_summary_deduplicates_camera_views_and_counts_completed_frames():
    direct = {"uuid": "first", "camera": "front", "frame": 7, "smpl_valid": True, "pose_source": "4d_humans"}
    rows = [
        direct, {**direct, "camera": "front_right"},
        {**direct, "frame": 8, "pose_source": "omnire_interpolation", "limbs_visible": False},
        {**direct, "uuid": "second", "camera": "back"},
        {**direct, "uuid": "unselected"},
        {**direct, "frame": 9, "smpl_valid": False},
    ]
    assert front_pose_summary({"selected_uuids": ["first", "second"], "observations": rows}) == {
        "tracks": 1, "direct_frames": 1, "completed_frames": 1,
    }
