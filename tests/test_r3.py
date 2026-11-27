import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from research.deformable import OmniReDeformationField, OmniRePedestrianGaussians, PedestrianGaussians
from research.ground import fit_road_plane
from research.pose_constraints import PoseConstraints
from research.smpl_nodes import SMPLTrack


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


def test_deformation_moves_local_regions_and_backpropagates():
    means = torch.tensor([[-0.2, 0., -0.5], [0.2, 0., -0.5], [0., 0., 0.4]])
    seed = {"means": means, "quats": torch.tensor([[1., 0., 0., 0.]]).expand(3, 4),
            "scales": torch.full_like(means, -3), "opacities": torch.zeros(3, 1),
            "features_dc": torch.zeros_like(means), "features_rest": torch.zeros(3, 13)}
    time = torch.tensor(0.05)
    box = torch.eye(4)
    box[:3, 3] = torch.tensor([2., 3., 0.])
    node = PedestrianGaussians(0, seed, torch.tensor([0., 0.1, 0.2]), torch.tensor([True] * 3),
                                torch.tensor([0.8, 0.8, 1.7]), True)
    original = node.get_gaussians_at(time, box, torch.zeros(6), True)
    assert torch.allclose(original["means"], means + box[:3, 3])
    with torch.no_grad():
        node.offsets[1, 2, 1] = 0.5
    updated = node.get_gaussians_at(time, box, torch.zeros(6), True)
    assert torch.allclose(updated["means"].mean(0), original["means"].mean(0))
    assert updated["means"][0, 1] > original["means"][0, 1]
    assert updated["means"][1, 1] < original["means"][1, 1]
    assert updated["velocities"][0, 1] > 0
    assert not node.get_gaussians_at(time, box, torch.zeros(6), False)["opacities"].any()
    updated["means"][0].sum().backward()
    assert torch.isfinite(node.offsets.grad).all() and node.offsets.grad.abs().sum() > 0


def test_contact_prefit_respects_swing_foot_and_propagates_gradient():
    vertices = torch.tensor([[-0.20, 0., 0.08], [-0.16, 0.08, 0.08], [-0.20, 0.12, 0.08],
                             [0.20, 0., 0.39], [0.16, 0.08, 0.39], [0.20, 0.12, 0.39]])
    weights = torch.nn.functional.one_hot(torch.tensor([10, 10, 10, 11, 11, 11]), 24).float()
    body = SimpleNamespace(parents=(-1,) + (0,) * 23, faces=torch.tensor([[0, 1, 2], [3, 4, 5]]), weights=weights,
                           canonical=lambda _: (vertices, torch.zeros(24, 3), torch.eye(4).repeat(24, 1, 1)))
    identity = [1, 0, 0, 0]
    track = {"uuid": "test", "betas": [0] * 10, "frames": [
        {"valid": True, "frame_idx": i, "timestamp_s": i / 10, "root_translation": [0.02 * i, 0, 0],
         "root_quat_wxyz": identity, "body_quats_wxyz": [identity] * 23} for i in range(3)]}
    person = SMPLTrack(track, body, torch.eye(4), 0, 13, 0.99, True)
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
    assert losses["foot_contact"] > 0 and losses["foot_slip"] > 0
    sum(losses.values()).backward()
    assert torch.isfinite(person.translation.grad).all() and person.translation.grad[1, 2] > 0


@pytest.fixture
def pedestrian_model_class(monkeypatch):
    class Base(torch.nn.Module):
        def step_post_backward(self, step):
            self.refined = (step, len(self.gauss_params["means"]))

        def scene_gaussians(self, time):
            return self.base_packet

    dependency = ModuleType("nerfstudio.models.splatad")
    dependency.SplatADModel = Base
    dependency.SplatADModelConfig = object
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    path = Path(__file__).resolve().parents[1] / "research/model.py"
    spec = importlib.util.spec_from_file_location("research._r3_test_model", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.PedestrianModel


def test_mcmc_callback_and_resizing_leave_pedestrian_parameters_intact(pedestrian_model_class):
    model = pedestrian_model_class()
    model.gauss_params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.zeros(2, 3)),
        "id": torch.nn.Parameter(torch.zeros(2, 1), requires_grad=False),
    })
    model.people = torch.nn.ModuleList([torch.nn.Linear(3, 3)])
    original = model.people[0].weight
    state = model.state_dict()
    state["gauss_params.means"] = torch.ones(5, 3)
    state["gauss_params.id"] = torch.ones(5, 1)
    model.step_post_backward(600)
    assert not hasattr(model, "refined")
    model.refinement_enabled = True
    model.load_state_dict(state)
    model.step_post_backward(600)
    assert model.refined == (600, 5)
    assert model.people[0].weight is original
    assert not model.gauss_params["id"].requires_grad


def test_omnire_uses_r3_adapter_and_restores_warmup_from_checkpoint(pedestrian_model_class):
    means = torch.tensor([[-0.2, 0., -0.5], [0.2, 0., -0.5]])
    seed = {"means": means, "quats": torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
            "scales": torch.full_like(means, -3), "opacities": torch.zeros(2, 1),
            "features_dc": torch.ones_like(means), "features_rest": torch.ones(2, 13)}
    time = torch.tensor(10.1)
    times = torch.tensor([10., 10.1, 10.2])
    size = torch.tensor([0.8, 0.8, 1.7])
    box = torch.tensor([[0., -1., 0., 3.], [1., 0., 0., 2.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    velocity = torch.tensor([2., 0., 0., 0., 0., 0.])
    presence = torch.tensor([True, False])

    def build_model():
        model = pedestrian_model_class()
        model.refinement_enabled = False
        model.config = SimpleNamespace(far_deform_start=100)
        model.people = torch.nn.ModuleList()
        model.other_people = torch.nn.ModuleList(
            OmniRePedestrianGaussians(i, seed, times, torch.ones(3, dtype=torch.bool), size) for i in range(2)
        )
        model.far_deformation_field = OmniReDeformationField()
        model.dynamic_actors = SimpleNamespace(get_boxes2world=lambda *_args, **_kwargs: (box.repeat(2, 1, 1), presence),
                                               get_velocities=lambda _: velocity.repeat(2, 1))
        with torch.no_grad():
            model.base_packet = model.other_people[0].get_gaussians_at(time, box, velocity, True)
        return model

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = build_model()
        restored = build_model().eval()
    model.update_to_step(99)
    rigid = model.scene_gaussians(time)
    assert torch.equal(rigid["means"][2:], rigid["means"][:2])
    assert not model.far_bound_losses
    model.update_to_step(100)
    output = model.scene_gaussians(time)
    assert len(output["means"]) == 4  # Absent track adds no points to the renderer.
    assert not torch.allclose(output["means"][2:], rigid["means"][2:])
    assert not torch.allclose(output["quats"][2:], rigid["quats"][2:])
    assert torch.allclose(output["quats"][2:].norm(dim=-1), torch.ones(2))
    assert torch.equal(output["features_rest"], rigid["features_rest"])
    assert torch.isfinite(output["velocities"]).all()
    assert not torch.allclose(output["velocities"][2:], rigid["velocities"][2:])
    assert len(model.far_bound_losses) == 1

    (output["means"][2:].sum() + output["quats"][2:, 1].sum()).backward()
    person = model.other_people[0]
    assert person.means.grad is None
    assert person.embedding.grad.abs().sum() > 0
    assert model.other_people[1].embedding.grad is None
    assert model.far_deformation_field.position_delta.weight.grad.abs().sum() > 0
    assert model.far_deformation_field.rotation_delta.weight.grad.abs().sum() > 0

    restored.update_to_step(100)
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        replay = restored.scene_gaussians(time)
    for key in output:
        torch.testing.assert_close(output[key], replay[key])
