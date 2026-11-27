"""CPU checks for R3: OmniRe regularizers, deformable pedestrians, ground contact and bank routing."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from research.deformable import DeformablePedestrian, OmniReDeformationField
from research.ground import fit_road_plane
from research.pose_constraints import PoseConstraints
from research.smpl_nodes import SMPLTrack
from research.smpl_regularization import SMPLRegularizer

IDENTITY = [1, 0, 0, 0]
ALL_WEIGHTS = dict.fromkeys(("trans_smooth", "joint_smooth", "local_offset", "knn", "max_scale", "sharp_shape"), 1.0)


def toy_body():
    vertices = torch.tensor([[-0.20, 0., 0.08], [-0.16, 0.08, 0.08], [-0.20, 0.12, 0.08],
                             [0.20, 0., 0.39], [0.16, 0.08, 0.39], [0.20, 0.12, 0.39]])
    weights = torch.nn.functional.one_hot(torch.tensor([10, 10, 10, 11, 11, 11]), 24).float()
    joints = torch.zeros(24, 3)
    joints[4] = torch.tensor([0.1, 0., 0.5])
    return SimpleNamespace(parents=(-1,) + (0,) * 23, faces=torch.tensor([[0, 1, 2], [3, 4, 5]]),
                           weights=weights, vertices=vertices,
                           canonical=lambda _: (vertices, joints, torch.eye(4).repeat(24, 1, 1)))


def toy_track(uuid="person", roots=None, frames=11):
    roots = roots if roots is not None else [[0.02 * i, 0, 0] for i in range(frames)]
    return {"uuid": uuid, "betas": [0] * 10, "frames": [
        {"valid": True, "frame_idx": i, "timestamp_s": i / 10, "root_translation": root,
         "root_quat_wxyz": IDENTITY, "body_quats_wxyz": [IDENTITY] * 23} for i, root in enumerate(roots)]}


def test_ground_plane_is_local_and_allows_slopes():
    rng = np.random.default_rng(42)
    center = np.array([1000., -200., 9.])
    xy = rng.uniform(-1, 1, size=(160, 2))
    height = 8.1 + xy @ np.array([0.07, -0.04]) + rng.normal(0, 0.008, 160)
    height[:18] += 0.25
    points = np.column_stack((xy + center[:2], height))
    normal_x, normal_y, normal_z, offset = fit_road_plane(points, center, 8.1)
    assert np.linalg.norm([normal_x, normal_y, normal_z]) == pytest.approx(1)
    assert normal_z > 0.9
    assert abs(normal_x * center[0] + normal_y * center[1] + normal_z * 8.1 + offset) < 0.01
    assert fit_road_plane(points[:5], center, 8.1) is None


def test_contact_prefit_respects_swing_foot_and_propagates_gradient():
    person = SMPLTrack(toy_track(frames=3), toy_body(), torch.eye(4), 0, 13, 0.99, True)
    planes = {str(i): [0, 0, 1, 0] for i in range(3)}
    person.constraints = PoseConstraints(person, {0, 1, 2}, planes, torch.eye(4), prefit=True)
    time = torch.tensor(0.1)
    measured = person.constraints.measure(person, time)
    assert measured["left_foot_stance"] and not measured["right_foot_stance"]
    assert measured["left_foot_height_cm"] == pytest.approx(0, abs=1e-4)
    assert measured["right_foot_height_cm"] == pytest.approx(31, abs=1e-4)
    with torch.no_grad():
        person.translation[1, 2] += 0.05
    losses = person.constraints.losses(person, time)
    assert set(losses) == {"foot_contact", "ground_penetration", "foot_slip"}
    assert losses["foot_contact"] > 0 and losses["foot_slip"] > 0
    sum(losses.values()).backward()
    assert torch.isfinite(person.translation.grad).all() and person.translation.grad[1, 2] > 0


def test_omnire_smoothing_ignores_steady_walk_and_pulls_back_a_jitter():
    steady = SMPLTrack(toy_track(), toy_body(), torch.eye(4), 0, 13, 0.99, True)
    regularizer = SMPLRegularizer(steady)
    losses = regularizer.losses(steady, torch.tensor(0.5), 10, ALL_WEIGHTS)
    assert losses["trans_smooth"].item() == pytest.approx(0, abs=1e-6)
    assert losses["joint_smooth"].item() == pytest.approx(0, abs=1e-6)
    assert {"knn", "max_scale", "sharp_shape"} <= set(losses)

    roots = [[0.02 * i, 0, 0] for i in range(11)]
    roots[5][1] = 0.1
    jitter = SMPLTrack(toy_track(roots=roots), toy_body(), torch.eye(4), 0, 13, 0.99, True)
    losses = SMPLRegularizer(jitter).losses(jitter, torch.tensor(0.5), 11, ALL_WEIGHTS)
    assert losses["trans_smooth"] > 0 and losses["joint_smooth"] > 0
    (losses["trans_smooth"] + losses["joint_smooth"]).backward()
    # Gradient descent moves the outlier back towards its neighbours.
    assert jitter.translation.grad[5, 1] > 0
    # Neighbours are detached in OmniRe's translation term; joint smoothing still reaches them.
    assert "sharp_shape" not in losses


def test_regularizer_skips_times_between_track_segments():
    track = toy_track(frames=4)
    track["frames"][2]["frame_idx"] = 7
    track["frames"][2]["timestamp_s"] = 0.7
    track["frames"][3]["frame_idx"] = 8
    track["frames"][3]["timestamp_s"] = 0.8
    person = SMPLTrack(track, toy_body(), torch.eye(4), 0, 13, 0.99, True)
    assert SMPLRegularizer(person).losses(person, torch.tensor(0.4), 10, ALL_WEIGHTS) == {}
    losses = SMPLRegularizer(person).losses(person, torch.tensor(0.1), 10, ALL_WEIGHTS)
    assert "joint_smooth" not in losses


def deformable_seed(count=3):
    means = torch.tensor([[-0.2, 0., -0.5], [0.2, 0., -0.5], [0., 0., 0.4]])[:count]
    return {"means": means, "quats": torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1),
            "scales": torch.full_like(means, -3), "opacities": torch.zeros(count, 1),
            "features_dc": torch.ones_like(means), "features_rest": torch.ones(count, 13)}


def test_deformable_pedestrian_is_rigid_before_warmup_and_follows_omnire_after():
    seed = deformable_seed()
    node = DeformablePedestrian(0, seed, torch.tensor([10., 10.1, 10.2, 10.3]), torch.tensor([0.8, 0.8, 1.7]))
    box = torch.tensor([[0., -1., 0., 3.], [1., 0., 0., 2.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    velocity = torch.tensor([2., 0., 0., 0., 0., 0.])
    time = torch.tensor(10.1)
    rigid, outside = node.get_gaussians_at(time, box, velocity, torch.tensor(True))
    assert outside is None
    torch.testing.assert_close(rigid["means"], seed["means"] @ box[:3, :3].T + box[:3, 3])
    torch.testing.assert_close(rigid["velocities"], velocity[:3].expand(3, 3))

    torch.manual_seed(0)
    field = OmniReDeformationField()
    deformed, outside = node.get_gaussians_at(time, box, velocity, torch.tensor(True), field)
    assert not torch.allclose(deformed["means"], rigid["means"])
    assert torch.isfinite(deformed["velocities"]).all() and outside is not None
    (deformed["means"].sum() + deformed["quats"][:, 1].sum() + outside).backward()
    assert node.means.grad is None  # OmniRe stop_optimizing_canonical_xyz
    assert node.embedding.grad.abs().sum() > 0
    assert field.position_delta.weight.grad.abs().sum() > 0
    assert field.rotation_delta.weight.grad.abs().sum() > 0
    hidden, _ = node.get_gaussians_at(time, box, velocity, torch.tensor(False), field)
    assert not hidden["opacities"].any()


@pytest.fixture
def pedestrian_model_class(monkeypatch):
    class Base(torch.nn.Module):
        @property
        def id(self):
            return self.gauss_params["id"]

        def populate_modules(self):
            self.gauss_params = torch.nn.ParameterDict(
                {name: torch.nn.Parameter(value.clone(), requires_grad=name != "id")
                 for name, value in self.native_gaussians.items()})
            self.actor_seeds = [2, 2, 2, 2]
            self.step = 0

        def step_post_backward(self, step):
            self.refined = (step, len(self.gauss_params["means"]))

        def update_to_step(self, step):
            pass

        def get_param_groups(self):
            return {"means": [self.gauss_params["means"]]}

        def scene_gaussians(self, time):
            return self.base_packet

    dependency = ModuleType("nerfstudio.models.splatad")
    dependency.SplatADModel = Base
    dependency.SplatADModelConfig = object
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    path = Path(__file__).resolve().parents[1] / "research/model.py"
    spec = importlib.util.spec_from_file_location("research._r3_test_model", path, submodule_search_locations=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "research"
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class Boxes:
    """Fake SplatAD dynamic actors: boxes slide along x at 4 m/s plus a learnable offset per
    trajectory sample, linearly interpolated in time like DynamicActors.get_boxes2world."""

    def __init__(self, count):
        self.unique_timestamps = torch.arange(11) / 10
        self.actor_sizes = torch.ones(count, 3)
        self.offset = torch.zeros(len(self.unique_timestamps), count, 3, requires_grad=True)
        self.count = count

    def get_boxes2world(self, times, flatten=False):
        times = times.reshape(-1)
        grid = self.unique_timestamps
        hi = torch.searchsorted(grid, times).clamp(1, len(grid) - 1)
        lo = hi - 1
        alpha = ((times - grid[lo]) / (grid[hi] - grid[lo])).clamp(0, 1)[:, None, None]
        offset = self.offset[lo] * (1 - alpha) + self.offset[hi] * alpha
        shift = torch.zeros(len(times), self.count, 3)
        shift[..., 0] = times[:, None] * 4.0
        boxes = torch.eye(4).repeat(len(times), self.count, 1, 1).clone()
        boxes[..., :3, 3] = shift + offset
        return boxes, torch.ones(len(times), self.count, dtype=torch.bool)

    def get_velocities(self, time):
        return torch.zeros(1, self.count, 6)


def close_rows(uuid, frames, height=200, bin_name="near", source="4d_humans"):
    return [{"uuid": uuid, "frame": f, "camera": "front", "smpl_valid": True, "usable": True, "bin": bin_name,
             "bbox_height_px": height, "pose_source": source} for f in frames]


def build_model(module, monkeypatch, variant, observations=None):
    walk = torch.eye(4).repeat(4, 1, 1)
    walk[:, 0, 3] = torch.arange(4) * 0.4
    trajectories = [
        {"uuid": "car", "label": "Car", "stationary": False, "deformable": False, "poses": walk},
        {"uuid": "smpl", "label": "Pedestrian", "stationary": False, "deformable": True, "poses": walk},
        {"uuid": "walker", "label": "Pedestrian", "stationary": False, "deformable": True, "poses": walk},
        {"uuid": "idle", "label": "Pedestrian", "stationary": True, "deformable": True, "poses": torch.eye(4).repeat(4, 1, 1)},
    ]
    if observations is None:
        observations = close_rows("smpl", range(6))
    documents = {"manifest": {"selected_uuids": ["smpl"], "tracks_sha256": "tracks", "observations": observations},
                 "tracks": {"tracks": [toy_track("smpl")], "body_sha256": "body"}}
    monkeypatch.setattr(module, "read_json", documents.__getitem__)
    monkeypatch.setattr(module, "validate_tracks", lambda document, _scene: document)
    monkeypatch.setattr(module, "sha256", lambda path: path)
    monkeypatch.setattr(module, "BodyModel", lambda _path: toy_body())
    model = module.PedestrianModel()
    model.config = SimpleNamespace(
        variant=variant, scene_id="030", smpl_init_opacity=0.99, manifest_path="manifest", manifest_digest="manifest",
        tracks_path="tracks", body_path="body", feature_dim=13, ground_path=None, ground_prefit=True,
        deform_start=100, deform_max_gaussians=5000, deform_min_travel=0.5,
        trans_smooth_weight=0.01, joint_smooth_weight=0.01, knn_weight=0.001, max_scale_weight=0.05,
        sharp_shape_weight=1.0, out_of_bound_weight=1.0, contact_weight=0.1, slip_weight=0.1,
        smpl_follow_box=True, smpl_min_frames=5, smpl_min_box_px=80.0, local_offset_weight=0.01)
    model.kwargs = {"metadata": {"trajectories": trajectories, "scene_from_world": torch.eye(4),
                                 "scene_time_offset": 0.0}}
    model.dynamic_actors = Boxes(4)
    widths = {"means": 3, "quats": 4, "scales": 3, "opacities": 1, "features_dc": 3, "features_rest": 13}
    model.native_gaussians = {name: torch.arange(10 * width, dtype=torch.float32).reshape(10, width)
                              for name, width in widths.items()}
    model.native_gaussians["id"] = torch.tensor([[0.], [0.], [1.], [1.], [2.], [2.], [3.], [3.], [4.], [4.]])
    model.populate_modules()
    return model


def test_r2_routes_are_unchanged_and_bank_stays_fixed(pedestrian_model_class, monkeypatch):
    model = build_model(pedestrian_model_class, monkeypatch, "R2")
    assert [route["representation"] for route in model.actor_routes] == ["rigid", "smpl", "rigid", "rigid"]
    assert torch.equal(model.gauss_params["id"].flatten(), torch.tensor([0., 0., 2., 2., 3., 3., 4., 4.]))
    assert not model.refinement_enabled and not model.deformable_people and not model.smpl_regularizers
    model.step_post_backward(600)
    assert not hasattr(model, "refined")
    groups = model.get_param_groups()
    assert {"smpl_means", *pedestrian_model_class.POSE_GROUPS} <= set(groups)
    assert not any(name.startswith("deformable_") for name in groups)


def test_r3_moves_only_walking_pedestrians_and_refines_the_rest(pedestrian_model_class, monkeypatch):
    model = build_model(pedestrian_model_class, monkeypatch, "R3")
    assert [route["representation"] for route in model.actor_routes] == ["rigid", "smpl", "deformable", "rigid"]
    # Car, idle pedestrian and background stay in SplatAD's bank, with their native rows.
    assert torch.equal(model.gauss_params["id"].flatten(), torch.tensor([0., 0., 3., 3., 4., 4.]))
    torch.testing.assert_close(model.gauss_params["means"], model.native_gaussians["means"][[0, 1, 6, 7, 8, 9]])
    torch.testing.assert_close(model.deformable_people[0].means, model.native_gaussians["means"][[4, 5]])
    assert model.refinement_enabled and len(model.smpl_regularizers) == 1
    model.step = 600
    model.step_post_backward(600)
    assert model.refined == (600, 6)

    state = model.state_dict()
    state["gauss_params.means"] = torch.ones(9, 3)
    state["gauss_params.id"] = torch.ones(9, 1)
    for name in ("quats", "scales", "opacities", "features_dc", "features_rest"):
        state[f"gauss_params.{name}"] = torch.ones(9, model.gauss_params[name].shape[1])
    person = model.people[0].translation
    model.load_state_dict(state)
    assert len(model.gauss_params["means"]) == 9 and not model.gauss_params["id"].requires_grad
    assert model.people[0].translation is person

    groups = model.get_param_groups()
    assert {"deformable_means", "deformable_embedding", "deformable_network", "smpl_translation"} <= set(groups)

    model.last_actor_time = torch.tensor(0.5)
    model.deform_bounds = [torch.tensor(0.5)]
    losses = model.r3_losses()
    assert {"r3_trans_smooth", "r3_joint_smooth", "r3_knn", "r3_max_scale"} <= set(losses)
    assert float(losses["r3_out_of_bound"]) == pytest.approx(0.5)
    model.update_to_step(2500)
    assert model.step == 2500


def test_r3_sends_a_far_smpl_track_to_the_deformable_node(pedestrian_model_class, monkeypatch):
    far = close_rows("smpl", range(8), height=60, bin_name="far")
    model = build_model(pedestrian_model_class, monkeypatch, "R3", observations=far)
    assert not model.people
    assert [route["representation"] for route in model.actor_routes] == ["rigid", "deformable", "deformable", "rigid"]
    assert model.actor_routes[1]["smpl_rejected"] == "0 close direct frames"


def test_r3_smpl_follows_the_learned_box_and_feeds_it_gradients(pedestrian_model_class, monkeypatch):
    unanchored = SMPLTrack(toy_track(), toy_body(), torch.eye(4), 0, 13, 0.99, True)
    reference = {float(t): unanchored.mesh_at(torch.tensor(t))[0] for t in (0.0, 0.25, 0.5)}

    model = build_model(pedestrian_model_class, monkeypatch, "R3")
    person = model.people[0]
    assert person.box_at is not None and model.actor_routes[1]["follows_box"]
    # Anchoring re-expresses the pose and leaves the mesh where it was.
    for t, mesh in reference.items():
        torch.testing.assert_close(person.mesh_at(torch.tensor(t))[0], mesh, atol=1e-5, rtol=0)
        torch.testing.assert_close(person.mesh_at(torch.tensor(t), initial=True)[0], mesh, atol=1e-5, rtol=0)
    # When trajectory optimization moves the box, the mesh moves with it; the initial mesh does not.
    with torch.no_grad():
        model.dynamic_actors.offset[:, 1, 1] = 0.3
    moved = person.mesh_at(torch.tensor(0.25))[0]
    torch.testing.assert_close(moved - reference[0.25], torch.tensor([0., 0.3, 0.]).expand_as(moved), atol=1e-5, rtol=0)
    torch.testing.assert_close(person.mesh_at(torch.tensor(0.25), initial=True)[0], reference[0.25], atol=1e-5, rtol=0)
    output = person.get_gaussians_at(torch.tensor(0.25))
    output["means"][:, 1].sum().backward()
    assert model.dynamic_actors.offset.grad[:, 1, 1].sum() > 0


def test_r3_velocity_includes_box_motion_and_has_gradients(pedestrian_model_class, monkeypatch):
    model = build_model(pedestrian_model_class, monkeypatch, "R3")
    person = model.people[0]
    output = person.get_gaussians_at(torch.tensor(0.25))
    # Box slides at 4 m/s; the anchored root itself walks 0.02 m per 0.1 s inside the scene.
    torch.testing.assert_close(output["velocities"][:, 0], torch.full((6,), 0.2), atol=1e-3, rtol=0)
    output["velocities"].sum().backward()
    assert person.translation.grad is not None and person.translation.grad.abs().sum() > 0


def gradient_matrix(module, monkeypatch):
    """Back-propagate a render stand-in and every R3 loss on their own; report which groups move."""
    import random

    torch.manual_seed(0)
    random.seed(3)
    planes = {"smpl": {str(i): [0, 0, 1, 0] for i in range(11)}}
    monkeypatch.setattr(module.PedestrianModel, "ground_inputs",
                        lambda self, manifest: ({"smpl": set(range(11))}, planes))
    model = build_model(module, monkeypatch, "R3")
    person = model.people[0]
    with torch.no_grad():
        # Real training is never this clean: a jittered root, a sunken foot, a needle Gaussian
        # and an uneven learned box. Exact zeros would hide gradients behind |0|.
        person.translation[5] += torch.tensor([0.0, 0.04, -0.06])
        person.scales[0] = torch.tensor([-1.0, -5.0, -5.0])
        model.dynamic_actors.offset[:, 1] += 0.01 * torch.randn(11, 3)
    bank = model.gauss_params
    model.base_packet = {
        "means": bank["means"], "quats": bank["quats"], "scales": bank["scales"].exp(),
        "opacities": bank["opacities"].sigmoid().squeeze(-1), "features_dc": bank["features_dc"],
        "features_rest": bank["features_rest"], "velocities": torch.zeros_like(bank["means"]),
    }
    model.step = 200  # past deform_start=100, so OmniRe's MLP is active
    walker = model.deformable_people[0]
    groups = {
        "Gaussian nền/xe": [bank[name] for name in ("means", "quats", "scales", "opacities", "features_dc")],
        "Gaussian SMPL": [person.means, person.quats, person.scales, person.opacities, person.features_dc],
        "Pose SMPL": [person.translation, person.root, person.body],
        "Quỹ đạo hộp": [model.dynamic_actors.offset],
        "Gaussian biến dạng": [walker.quats, walker.scales, walker.opacities, walker.features_dc],
        "Tâm canonical biến dạng": [walker.means],
        "MLP biến dạng": [*model.deformation_field.parameters(), walker.embedding],
    }
    weights = None

    def render_stand_in():
        nonlocal weights
        packet = model.scene_gaussians(torch.tensor(0.25))
        if weights is None:
            weights = {key: torch.randn_like(value) for key, value in packet.items()}
        return sum((weights[key] * value).sum() for key, value in packet.items())

    def touched(loss):
        for params in groups.values():
            for p in params:
                p.grad = None
        loss.backward(retain_graph=True)
        return {name for name, params in groups.items()
                if any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)}

    rows = {"render (RGB + LiDAR)": touched(render_stand_in())}
    model.scene_gaussians(torch.tensor(0.5))  # sets last_actor_time and the out-of-bound term
    for name, value in model.r3_losses().items():
        rows[name] = touched(value)
        model.scene_gaussians(torch.tensor(0.5))
    return rows, list(groups)


def test_r3_gradients_reach_mesh_mlp_gaussians_and_box(pedestrian_model_class, monkeypatch):
    rows, groups = gradient_matrix(pedestrian_model_class, monkeypatch)
    print("\n" + " | ".join(["loss", *groups]))
    for loss, reached in rows.items():
        print(" | ".join([loss, *("x" if group in reached else "." for group in groups)]))
    assert rows["render (RGB + LiDAR)"] == {"Gaussian nền/xe", "Gaussian SMPL", "Pose SMPL", "Quỹ đạo hộp",
                                           "Gaussian biến dạng", "MLP biến dạng"}
    # stop_optimizing_canonical_xyz: after the warm-up only the MLP moves deformable positions.
    assert "Tâm canonical biến dạng" not in rows["render (RGB + LiDAR)"]
    motion = {"Pose SMPL", "Quỹ đạo hộp"}
    assert rows["r3_trans_smooth"] == rows["r3_joint_smooth"] == motion
    assert rows["r3_local_offset"] == {"Pose SMPL"}
    assert rows["r3_foot_contact"] == rows["r3_ground_penetration"] == rows["r3_foot_slip"] == motion
    assert rows["r3_knn"] == rows["r3_max_scale"] == rows["r3_sharp_shape"] == {"Gaussian SMPL"}
    assert rows["r3_out_of_bound"] == {"MLP biến dạng"}
