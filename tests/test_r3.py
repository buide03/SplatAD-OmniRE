"""CPU checks for R3: OmniRe regularizers, far walkers on a borrowed gait, ground contact and bank routing."""
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from research.far_fit import FarFit, choose_phase
from research.gait import average_cycle, gait_cycle, local_mesh, sample_cycle
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


def toy_track(uuid="person", roots=None, frames=11, period=None, axis="x"):
    roots = roots if roots is not None else [[0.02 * i, 0, 0] for i in range(frames)]
    return {"uuid": uuid, "betas": [0] * 10, "frames": [
        {"valid": True, "frame_idx": i, "timestamp_s": i / 10, "root_translation": root,
         "root_quat_wxyz": IDENTITY, "body_quats_wxyz": stride_pose(i / 10, period, axis) if period else [IDENTITY] * 23}
        for i, root in enumerate(roots)]}


def about(degrees, axis="x"):
    half = np.radians(degrees) / 2
    quat = [float(np.cos(half)), 0.0, 0.0, 0.0]
    quat["xyz".index(axis) + 1] = float(np.sin(half))
    return quat


def about_x(degrees):
    return about(degrees, "x")


def stride_pose(time, period, axis="x"):
    swing = 20 * np.sin(2 * np.pi * time / period)
    pose = [IDENTITY] * 23
    pose[0], pose[1] = about(swing, axis), about(-swing, axis)
    pose[3], pose[4] = about(swing / 2 + 10, axis), about(-swing / 2 + 10, axis)
    return pose


SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)


def chain_body():
    """A z-up stick figure with SMPL's joint tree; one vertex sits on each joint and follows it."""
    joints = torch.tensor([
        [0, 0, .95], [0, .09, .9], [0, -.09, .9], [0, 0, 1.05], [0, .09, .5], [0, -.09, .5], [0, 0, 1.2],
        [0, .09, .08], [0, -.09, .08], [0, 0, 1.3], [.1, .09, .02], [.1, -.09, .02], [0, 0, 1.5],
        [0, .08, 1.45], [0, -.08, 1.45], [0, 0, 1.65], [0, .18, 1.45], [0, -.18, 1.45], [0, .2, 1.18],
        [0, -.2, 1.18], [0, .2, .92], [0, -.2, .92], [0, .2, .85], [0, -.2, .85]])
    faces = torch.tensor([[1, 4, 7], [2, 5, 8], [16, 18, 20], [17, 19, 21], [0, 3, 6]])
    return SimpleNamespace(parents=SMPL_PARENTS, faces=faces, weights=torch.eye(24), vertices=joints.clone(),
                           canonical=lambda _: (joints.clone(), joints.clone(), torch.eye(4).repeat(24, 1, 1)))


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


@pytest.fixture
def pedestrian_model_class(monkeypatch):
    class Base(torch.nn.Module):
        refinement_enabled = True  # splatad.patch default

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


def build_model(module, monkeypatch, variant, observations=None, seeds=None, donor_walks=True, evidence=None,
                library=None):
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
                 "tracks": {"tracks": [toy_track("smpl", period=0.6 if donor_walks else None)], "body_sha256": "body"}}
    monkeypatch.setattr(module, "read_json", documents.__getitem__)
    monkeypatch.setattr(module, "validate_tracks", lambda document, _scene: document)
    monkeypatch.setattr(module, "sha256", lambda path: path)
    monkeypatch.setattr(module, "BodyModel", lambda _path: toy_body())
    model = module.PedestrianModel()
    model.config = SimpleNamespace(
        variant=variant, scene_id="030", smpl_init_opacity=0.99, manifest_path="manifest", manifest_digest="manifest",
        tracks_path="tracks", body_path="body", feature_dim=13, ground_path=None, ground_prefit=True,
        gait_min_travel=0.5,
        trans_smooth_weight=0.01, joint_smooth_weight=0.01, knn_weight=0.001, max_scale_weight=0.05,
        sharp_shape_weight=1.0, contact_weight=0.1, slip_weight=0.1,
        smpl_follow_box=True, smpl_min_frames=5, smpl_min_box_px=80.0, local_offset_weight=0.01,
        deform_seed_path=None, deform_seed_digest="", deform_seed_min_points=64, init_scale=0.2,
        keypoints_path=None, keypoints_digest="", keypoint_min_score=0.3, keypoint_weight=0.05,
        lidar_fit_weight=0.05, lidar_fit_frames=4, gait_phase_steps=24, gait_library=library, far_ground=True)
    if seeds is not None:
        monkeypatch.setattr(module.PedestrianModel, "deform_seeds", lambda self: seeds)
    if evidence is not None:
        monkeypatch.setattr(module.PedestrianModel, "far_evidence", lambda self: evidence)
    model.kwargs = {"metadata": {"trajectories": trajectories, "scene_from_world": torch.eye(4),
                                 "scene_time_offset": 0.0}}
    model.dynamic_actors = Boxes(4)
    widths = {"means": 3, "quats": 4, "scales": 3, "opacities": 1, "features_dc": 3, "features_rest": 13}
    model.native_gaussians = {name: torch.arange(10 * width, dtype=torch.float32).reshape(10, width)
                              for name, width in widths.items()}
    model.native_gaussians["id"] = torch.tensor([[0.], [0.], [1.], [1.], [2.], [2.], [3.], [3.], [4.], [4.]])
    model.populate_modules()
    return model


def test_r2_routes_are_unchanged_and_bank_refines_like_splatad(pedestrian_model_class, monkeypatch):
    model = build_model(pedestrian_model_class, monkeypatch, "R2")
    assert [route["representation"] for route in model.actor_routes] == ["rigid", "smpl", "rigid", "rigid"]
    assert torch.equal(model.gauss_params["id"].flatten(), torch.tensor([0., 0., 2., 2., 3., 3., 4., 4.]))
    assert model.refinement_enabled and len(model.people) == 1 and not model.smpl_regularizers
    model.step = 600
    model.step_post_backward(600)
    assert model.refined == (600, 8)
    groups = model.get_param_groups()
    assert {"smpl_means", *pedestrian_model_class.POSE_GROUPS} <= set(groups)


def test_r3_moves_only_walking_pedestrians_and_refines_the_rest(pedestrian_model_class, monkeypatch):
    model = build_model(pedestrian_model_class, monkeypatch, "R3")
    assert [route["representation"] for route in model.actor_routes] == ["rigid", "smpl", "smpl_gait", "rigid"]
    # Car, idle pedestrian and background stay in SplatAD's bank, with their native rows.
    assert torch.equal(model.gauss_params["id"].flatten(), torch.tensor([0., 0., 3., 3., 4., 4.]))
    torch.testing.assert_close(model.gauss_params["means"], model.native_gaussians["means"][[0, 1, 6, 7, 8, 9]])
    assert model.refinement_enabled and len(model.smpl_regularizers) == len(model.people) == 2
    assert model.person_index == {"smpl": 0, "walker": 1}
    route = model.actor_routes[2]
    assert route["gait_donors"] == ["smpl"] and route["gait_period_s"] == pytest.approx(0.6)
    assert route["gait_phase"] == 0 and "keypoint_error_px" not in route  # No evidence, no search.
    assert route["rigid_gaussians_replaced"] == 2 and route["gaussians"] == 6
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
    assert {"smpl_means", "smpl_translation", "smpl_body_rotation"} <= set(groups)
    assert not any(name.startswith("deformable_") for name in groups)
    assert len(groups["smpl_body_rotation"]) == 2  # The far walker's pose trains like a near one.

    model.last_actor_time = torch.tensor(0.5)
    losses = model.r3_losses()
    assert {"r3_trans_smooth", "r3_joint_smooth", "r3_knn", "r3_max_scale"} <= set(losses)
    assert "r3_out_of_bound" not in losses


def test_r3_far_walker_swings_its_legs_with_the_borrowed_cycle(pedestrian_model_class, monkeypatch):
    model = build_model(pedestrian_model_class, monkeypatch, "R3")
    donor, walker = model.people
    # The donor's hips swing +-20 degrees every 0.6 s; the walker's box moves at the donor's speed,
    # so it replays the cycle at the same cadence, starting where its own path starts.
    hips = walker.initial_body[:, 0]
    angles = 2 * torch.atan2(hips[:, 1], hips[:, 0]).rad2deg()
    assert angles.max() - angles.min() > 30
    torch.testing.assert_close(walker.body, walker.initial_body)
    route = model.actor_routes[2]
    assert route["gait_cadence_hz"] == pytest.approx(1 / 0.6, abs=0.02) and route["gait_turn_deg"] == pytest.approx(0)
    # The stance foot stands on the box floor and the pelvis is centred in the box.
    mesh = local_mesh(walker, walker.root, walker.body, walker.translation).detach()
    assert float(mesh[..., 2].amin(-1).median()) == pytest.approx(-0.5, abs=1e-4)
    pelvis = walker.joints[0] + walker.translation
    torch.testing.assert_close(pelvis[:, :2].mean(0), torch.zeros(2), atol=1e-5, rtol=0)
    # It renders through the learned box like every R3 SMPL track.
    output = walker.get_gaussians_at(torch.tensor(0.25))
    output["means"][:, 1].sum().backward()
    assert model.dynamic_actors.offset.grad[:, 2, 1].sum() > 0


def test_gait_cycle_finds_the_period_and_turns_with_the_walker():
    person = SMPLTrack(toy_track(frames=31, period=1.0), toy_body(), torch.eye(4), 0, 13, 0.99, True)
    rotations = torch.eye(3).repeat(31, 1, 1)
    positions = torch.zeros(31, 3)
    positions[:, 0] = torch.arange(31) * 0.14  # 1.4 m/s along the box's own x axis
    cycle = gait_cycle(person, rotations, positions, height=1.75)
    assert cycle["period_s"] == pytest.approx(1.0) and cycle["stride_per_height"] == pytest.approx(0.8, abs=0.03)
    assert cycle["closing_deg"] < 1 and cycle["swing_deg"] > 10
    torch.testing.assert_close(cycle["direction"], torch.tensor([1.0, 0.0]))
    # Half a cycle later the legs have swapped.
    root, body, _ = sample_cycle(cycle, torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(body[0, 0], body[1, 1], atol=1e-5, rtol=0)
    # A track that only stands has no cycle to lend.
    standing = SMPLTrack(toy_track(frames=31), toy_body(), torch.eye(4), 0, 13, 0.99, True)
    assert gait_cycle(standing, rotations, positions, height=1.75) is None


def test_r3_without_a_walking_donor_leaves_far_walkers_rigid(pedestrian_model_class, monkeypatch):
    model = build_model(pedestrian_model_class, monkeypatch, "R3", donor_walks=False)
    assert [route["representation"] for route in model.actor_routes] == ["rigid", "smpl", "rigid", "rigid"]
    assert model.actor_routes[2]["gait_skipped"] == "no near SMPL walker with a closed leg cycle and no gait library"


def test_r3_without_any_near_smpl_track_has_nothing_to_borrow(pedestrian_model_class, monkeypatch):
    far = close_rows("smpl", range(8), height=60, bin_name="far")
    with pytest.raises(ValueError, match="no usable SMPL track"):
        build_model(pedestrian_model_class, monkeypatch, "R3", observations=far)


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
    walker = model.people[1]
    with torch.no_grad():
        walker.translation[5] += torch.tensor([0.02, -0.03, 0.01])
    groups = {
        "Gaussian nền/xe": [bank[name] for name in ("means", "quats", "scales", "opacities", "features_dc")],
        "Gaussian SMPL": [person.means, person.quats, person.scales, person.opacities, person.features_dc],
        "Pose SMPL": [person.translation, person.root, person.body],
        "Quỹ đạo hộp": [model.dynamic_actors.offset],
        "Gaussian người xa": [walker.means, walker.quats, walker.scales, walker.opacities, walker.features_dc],
        "Pose người xa": [walker.translation, walker.root, walker.body],
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
    model.scene_gaussians(torch.tensor(0.5))  # sets last_actor_time
    for name, value in model.r3_losses().items():
        rows[name] = touched(value)
        model.scene_gaussians(torch.tensor(0.5))
    return rows, list(groups)


def test_r3_gradients_reach_meshes_poses_and_box(pedestrian_model_class, monkeypatch):
    rows, groups = gradient_matrix(pedestrian_model_class, monkeypatch)
    print("\n" + " | ".join(["loss", *groups]))
    for loss, reached in rows.items():
        print(" | ".join([loss, *("x" if group in reached else "." for group in groups)]))
    assert rows["render (RGB + LiDAR)"] == set(groups)
    # Smoothing reaches both walkers.
    assert rows["r3_trans_smooth"] == rows["r3_joint_smooth"] == {"Pose SMPL", "Pose người xa", "Quỹ đạo hộp"}
    assert rows["r3_local_offset"] == {"Pose SMPL", "Pose người xa"}
    # Ground contact: the near track on its LiDAR planes, the far walker on its cuboid floor. The toy
    # walker's box slides at 4 m/s, too fast for a planted foot, and it starts on its floor, so none of the
    # three terms has a gradient for it here. test_far_walker_stands_on_its_cuboid_floor covers it.
    assert rows["r3_foot_contact"] == rows["r3_ground_penetration"] == rows["r3_foot_slip"] == {"Pose SMPL", "Quỹ đạo hộp"}
    assert rows["r3_knn"] == rows["r3_max_scale"] == {"Gaussian SMPL", "Gaussian người xa"}
    assert "r3_out_of_bound" not in rows


def dataset_seed(count, lidar, colour=(200, 30, 20)):
    generator = np.random.default_rng(4)
    points = (generator.random((count, 3)) - 0.5).astype(np.float32) * 1.2
    colors = np.tile(np.array([colour], np.uint8), (count, 1))
    return points, colors, np.r_[np.zeros(lidar, np.uint8), np.ones(count - lidar, np.uint8)]


def test_r3_paints_the_far_walker_from_the_dataset(pedestrian_model_class, monkeypatch):
    points, colors, source = dataset_seed(1000, 300)
    model = build_model(pedestrian_model_class, monkeypatch, "R3", seeds={"walker": (points, colors, source)})
    route = model.actor_routes[2]
    assert (route["colour_seed_points"], route["colour_seed_lidar"]) == (1000, 300)
    assert 0 < route["colour_matched"] <= 1
    torch.testing.assert_close(model.people[1].features_dc.detach(),
                               torch.tensor([[200., 30., 20.]]).expand(6, 3) / 255)
    # The near SMPL track keeps the default grey.
    torch.testing.assert_close(model.people[0].features_dc.detach(), torch.full((6, 3), 0.5))


def test_r3_keeps_grey_far_walkers_when_the_dataset_has_too_few_points(pedestrian_model_class, monkeypatch):
    model = build_model(pedestrian_model_class, monkeypatch, "R3", seeds={"walker": dataset_seed(10, 10)})
    assert "colour_seed_points" not in model.actor_routes[2]
    torch.testing.assert_close(model.people[1].features_dc.detach(), torch.full((6, 3), 0.5))


def test_mean_cycle_aligns_donors_that_start_at_different_phases_and_directions():
    rotations = torch.eye(3).repeat(31, 1, 1)
    forward = torch.zeros(31, 3)
    forward[:, 0] = torch.arange(31) * 0.14
    sideways = torch.zeros(31, 3)
    sideways[:, 1] = torch.arange(31) * 0.14  # This donor walks along its box's +y.
    early = SMPLTrack(toy_track("early", frames=31, period=1.0), toy_body(), torch.eye(4), 0, 13, 0.99, True)
    late_track = toy_track("late", frames=31, period=1.0)
    for i, frame in enumerate(late_track["frames"]):
        frame["body_quats_wxyz"] = stride_pose(i / 10 + 0.3, 1.0)
    late = SMPLTrack(late_track, toy_body(), torch.eye(4), 0, 13, 0.99, True)
    cycles = [gait_cycle(early, rotations, forward, 1.75), gait_cycle(late, rotations, sideways, 1.75)]
    mean = average_cycle(cycles)
    assert mean["donors"] == [c["uuid"] for c in sorted(cycles, key=lambda c: c["score"])]
    torch.testing.assert_close(mean["direction"], torch.tensor([1.0, 0.0]))
    # Aligned in phase, the average keeps the full swing instead of blurring two offset cycles.
    phases = torch.arange(24) / 24
    reference = sample_cycle(cycles[0] if cycles[0]["score"] <= cycles[1]["score"] else cycles[1], phases)[1]
    hips = 2 * torch.atan2(mean["body"][:, 0, 1], mean["body"][:, 0, 0]).rad2deg()
    assert hips.max() - hips.min() > 35
    assert float(quat_angle_deg(mean["body"][:, 0], reference[:, 0]).max()) < 6


def quat_angle_deg(a, b):
    dot = (a * b).sum(-1).abs().clamp(max=1)
    return torch.rad2deg(2 * dot.acos())


def side_camera(origin=(0.0, -10.0, 1.0)):
    """A pinhole camera looking along world +y: camera x = world x, camera y = world -z."""
    return {"camera_rotation": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            "camera_origin": list(origin), "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0}}


def far_walker(times, speed=1.4):
    """An anchored stick-figure walker whose box slides along x, and that box."""
    track = toy_track("far", frames=len(times))
    walker = SMPLTrack(track, chain_body(), torch.eye(4), 0, 13, 0.99, True)

    def box_at(query):
        query = query.reshape(-1)
        position = torch.zeros(len(query), 3)
        position[:, 0] = query * speed
        return torch.eye(3).repeat(len(query), 1, 1), position

    walker.anchor_to(box_at)
    return walker, box_at


def walking_donor_cycle():
    donor = SMPLTrack(toy_track("donor", frames=31, period=1.0, axis="y"), chain_body(), torch.eye(4), 0, 13, 0.99, True)
    positions = torch.zeros(31, 3)
    positions[:, 0] = torch.arange(31) * 0.14
    return average_cycle([gait_cycle(donor, torch.eye(3).repeat(31, 1, 1), positions, 1.75)])


def evidence_from(walker, times, height=1.75):
    """Keypoints a side camera would see and the LiDAR returns of each frame, for the walker as posed."""
    from research.far_fit import COCO_TO_SMPL

    camera = side_camera()
    rotation = torch.tensor(camera["camera_rotation"])
    origin = torch.tensor(camera["camera_origin"])
    joints = walker.joints_at(torch.arange(len(times))).detach()
    rows, sweeps = [], []
    for i, time in enumerate(times.tolist()):
        local = (joints[i] - origin) @ rotation
        pixel = local[:, :2] / local[:, 2:] * 1000 + torch.tensor([960.0, 540.0])
        keypoints = torch.zeros(17, 2)
        for coco, smpl in COCO_TO_SMPL:
            keypoints[coco] = pixel[smpl]
        scores = [0.0] * 5 + [0.9] * 12
        top, bottom = float(pixel[:, 1].min()), float(pixel[:, 1].max())
        rows.append({"timestamp_s": time, "keypoints_xy": keypoints.tolist(), "scores": scores,
                     "bbox_xyxy": [0.0, top, 10.0, bottom], **camera})
        mesh = local_mesh(walker, walker.root[i], walker.body[i], walker.translation[i], walker.means).detach()
        sweeps.append((i, mesh[mesh[:, 2] > -height / 2 + 0.1]))
    return rows, sweeps


def test_far_fit_finds_the_stride_phase_from_keypoints_and_lidar():
    times = torch.arange(15, dtype=torch.float32) / 10
    cycle = walking_donor_cycle()
    walker, box_at = far_walker(times)
    rotations, centers = box_at(times)
    truth = FarFit(walker, [], [], torch.eye(4), 0.0, times.tolist(), 1.75)
    choose_phase(walker, truth, cycle, times, rotations, centers, 1.75, steps=1)  # phase 0 without evidence
    from research.gait import gait_poses
    root, body, shift, _ = gait_poses(cycle, walker, times, rotations, centers, 1.75, phase=0.25)
    with torch.no_grad():
        walker.root.copy_(root), walker.body.copy_(body), walker.translation.copy_(shift)
    rows, sweeps = evidence_from(walker, times)

    fit = FarFit(walker, rows, sweeps, torch.eye(4), 0.0, times.tolist(), 1.75)
    assert fit.keypoint_observations == 15 and fit.lidar_frames == 15
    decided = choose_phase(walker, fit, cycle, times, rotations, centers, 1.75, steps=20)
    assert decided["gait_phase"] == pytest.approx(0.25)
    assert decided["keypoint_error_px"] == pytest.approx(0, abs=1e-2) and decided["lidar_fit"] < 1e-6
    torch.testing.assert_close(walker.initial_body, walker.body)

    # Away from the answer both terms are positive and their gradients pull the pose back.
    with torch.no_grad():
        walker.body[:, 0] = torch.tensor(about(35, "y"))
    keypoint, lidar = fit.keypoint_loss(walker), fit.lidar_loss(walker, frames=4)
    assert keypoint > 0 and lidar > 0
    (keypoint + lidar).backward()
    assert walker.body.grad[:, 0].abs().sum() > 0
    step = walker.body.detach() - 0.05 * walker.body.grad / walker.body.grad.norm()
    with torch.no_grad():
        walker.body.copy_(step)
    assert fit.keypoint_loss(walker) < keypoint


def test_r3_adds_keypoint_and_lidar_terms_for_far_walkers(pedestrian_model_class, monkeypatch):
    camera = side_camera((0.0, -10.0, 0.0))
    keypoints = {"walker": [{"timestamp_s": 0.5, "keypoints_xy": [[960.0, 540.0]] * 17,
                             "scores": [0.0] * 5 + [0.9] * 12, "bbox_xyxy": [900.0, 500.0, 1000.0, 600.0],
                            **camera}]}
    sweeps = {"walker": [(5, np.array([[0.0, 0.0, 0.1], [0.05, 0.0, 0.3]], np.float32))]}
    model = build_model(pedestrian_model_class, monkeypatch, "R3",
                        evidence=(keypoints, sweeps, [i / 10 for i in range(11)]))
    route = model.actor_routes[2]
    assert route["keypoint_observations"] == 1 and route["lidar_frames"] == 1
    assert "keypoint_error_px" in route and "lidar_fit" in route and 0 <= route["gait_phase"] < 1
    assert list(model.far_fits) == ["1"]  # Only the far walker; the near SMPL track keeps 4D-Humans.
    model.last_actor_time = torch.tensor(0.5)
    losses = model.r3_losses()
    assert {"r3_keypoint", "r3_lidar_fit"} <= set(losses)
    (losses["r3_keypoint"] + losses["r3_lidar_fit"]).backward()
    walker = model.people[1]
    assert walker.body.grad is not None and walker.translation.grad.abs().sum() > 0
    assert model.people[0].body.grad is None


def test_far_walker_stands_on_its_cuboid_floor(pedestrian_model_class, monkeypatch):
    model = build_model(pedestrian_model_class, monkeypatch, "R3")
    walker = model.people[1]
    assert model.actor_routes[2]["ground"] == "cuboid_floor"
    # Box centres sit at z = 0 with height 1, so the floor is z = -0.5 at every pose.
    torch.testing.assert_close(walker.constraints.normals, torch.tensor([0.0, 0.0, 1.0]).expand(11, 3))
    torch.testing.assert_close(walker.constraints.offsets, torch.full((11,), 0.5))

    # A stick figure walking at 1.4 m/s plants its stance foot on the floor; lifted, it is pulled back.
    from research.gait import floor_planes
    times = torch.arange(15, dtype=torch.float32) / 10
    walker, box_at = far_walker(times)
    rotations, centers = box_at(times)
    choose_phase(walker, FarFit(walker, [], [], torch.eye(4), 0.0, [], 1.75), walking_donor_cycle(),
                 times, rotations, centers, 1.75, steps=1)
    constraints = PoseConstraints(walker, set(range(15)), floor_planes(walker, 1.75), torch.eye(4))
    assert int(constraints.contact.any(dim=-1).sum()) >= 10
    time = next(t for t, planted in zip(times, constraints.contact.any(dim=-1)) if planted)
    planted = float(constraints.losses(walker, time)["foot_contact"].detach())
    assert planted < 0.02  # Within a few centimetres: the floor is the median low point over the stride.
    with torch.no_grad():
        walker.translation[:, 2] += 0.1
    floating = constraints.losses(walker, time)["foot_contact"]
    assert floating > planted + 0.05
    floating.backward()
    assert walker.translation.grad[:, 2].sum() > 0


def test_scene_gait_is_saved_and_lent_to_a_scene_without_near_walkers(pedestrian_model_class, monkeypatch, tmp_path):
    library = tmp_path / "gait-library"
    donor_scene = build_model(pedestrian_model_class, monkeypatch, "R3", library=library)
    assert donor_scene.actor_routes[2]["gait_source"] == "scene"
    saved = json.loads((library / "030.json").read_text())
    assert saved["scene_id"] == "030" and saved["donors"] == ["smpl"] and len(saved["body"]) == 24

    # Another scene whose only SMPL track does not walk borrows the saved gait.
    (library / "030.json").rename(library / "028.json")
    saved["scene_id"] = "028"
    (library / "028.json").write_text(json.dumps(saved))
    borrowed = build_model(pedestrian_model_class, monkeypatch, "R3", donor_walks=False, library=library)
    route = borrowed.actor_routes[2]
    assert route["representation"] == "smpl_gait" and route["gait_source"] == "library"
    assert route["gait_donors"] == ["scene 028"] and route["gait_period_s"] == pytest.approx(0.6, abs=0.03)
    assert not (library / "030.json").exists()  # A scene without walkers saves nothing.
