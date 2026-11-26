"""CPU checks for the conventions shared by export, training and replay."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from research.compare import paired
from research.evaluate import crop_bounds
from research.geometry import apply, blend, joint_transforms, matrix_quat, quat_matrix, slerp
from research.prepare import complete_world_roots, cuboid_center_at, projected_box
from research.replay import camera_between, smooth_camera_poses
from research.smpl_nodes import SMPLTrack


IDENTITY = [1, 0, 0, 0]


def test_missing_root_is_interpolated_in_world_and_preserves_local_joints():
    from scipy.spatial.transform import Rotation

    matrices = np.broadcast_to(np.eye(3), (24, 3, 3)).copy()
    start = matrices.copy()
    end = matrices.copy()
    end[0] = Rotation.from_euler("z", 60, degrees=True).as_matrix()
    missing = matrices.copy()
    missing[0] = Rotation.from_euler("z", 160, degrees=True).as_matrix()
    missing[1] = Rotation.from_euler("x", 20, degrees=True).as_matrix()
    world_samples = {0: ("front_left", 0.0, start), 1: ("back", 0.05, missing), 2: ("front_right", 0.2, end)}
    complete_world_roots(world_samples, [True, False, True], [0.0, 0.1, 0.2])
    camera, time, corrected = world_samples[1]
    assert camera is None and time == 0.1
    assert np.allclose(corrected[0], Rotation.from_euler("z", 30, degrees=True).as_matrix())
    assert np.array_equal(corrected[1:], missing[1:])
    assert world_samples[0][2] is start and world_samples[2][2] is end


def test_cuboid_center_uses_capture_time_and_near_plane_boxes_are_retained():
    first, second = np.eye(4), np.eye(4)
    second[0, 3] = 0.2
    assert np.allclose(cuboid_center_at({0: first, 1: second}, [0.0, 0.1], 0, 0.05), [0.1, 0, 0])
    assert np.allclose(cuboid_center_at({0: first}, [0.0, 0.1], 0, 0.05), [0, 0, 0])

    camera = (np.eye(3), np.zeros(3), dict(fx=500, fy=500, cx=200, cy=200), 0.0)
    dimensions = np.array([0.2, 0.2, 0.4])
    assert np.allclose(projected_box([0, 0, 0.6], dimensions, 0, camera, 400, 400),
                       [100, 100, 300, 300], atol=1)
    assert projected_box([0, 0, -0.6], dimensions, 0, camera, 400, 400) is None


def person(train_pose=True, transform=None, frame_ids=(0, 1, 3), time_offset=1_600_000_000.0):
    vertices = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    joints = torch.zeros(24, 3)
    joints[0] = torch.tensor([0.0, 0.2, 0.0])
    joints[4] = torch.tensor([0.2, 0.0, 0.0])
    weights = torch.nn.functional.one_hot(torch.tensor([4, 0, 4]), 24).float()
    body = SimpleNamespace(
        parents=(-1,) + (0,) * 23, faces=torch.tensor([[0, 1, 2]]), weights=weights,
        canonical=lambda betas: (vertices, joints, torch.eye(4).repeat(24, 1, 1)),
    )
    track = {
        "uuid": "test-uuid", "betas": [0.0] * 10,
        "frames": [{"frame_idx": idx, "timestamp_s": time_offset + idx * 0.1, "valid": True,
                    "root_translation": [idx * 0.1, 0, 0], "root_quat_wxyz": IDENTITY,
                    "body_quats_wxyz": [IDENTITY] * 23} for idx in frame_ids],
    }
    return SMPLTrack(track, body, torch.eye(4) if transform is None else transform, time_offset, 13, 0.1, train_pose)


def test_lbs_reference_and_ellipsoid_orientation():
    track = person()
    q = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
    before = track.get_gaussians_at(torch.tensor(0.1))
    with torch.no_grad():
        track.body[1, 3] = q
    after = track.get_gaussians_at(torch.tensor(0.1))
    expected = (before["means"][0] - torch.tensor([0.3, 0., 0.])) @ quat_matrix(q).T + torch.tensor([0.3, 0., 0.])
    assert torch.allclose(after["means"][0], expected, atol=1e-6)
    assert torch.allclose(after["means"][1], before["means"][1], atol=1e-6)
    assert torch.allclose(quat_matrix(after["quats"][0]), quat_matrix(q) @ quat_matrix(before["quats"][0]), atol=1e-6)
    assert torch.all((after["scales"] > 0) & (after["scales"] <= 0.1))
    assert torch.equal(track.vertex_ids, torch.arange(3))


def test_time_precision_velocity_invalid_gap_and_gradients():
    track = person()
    midpoint = track.get_gaussians_at(torch.tensor(0.05))
    assert torch.allclose(midpoint["means"], track.means + torch.tensor([0.05, 0, 0]))
    assert torch.allclose(midpoint["velocities"], torch.tensor([[1., 0., 0.]]).expand(3, -1), atol=1e-5)
    assert not track.get_gaussians_at(torch.tensor(0.05), motion=False)["velocities"].any()
    assert not track.get_gaussians_at(torch.tensor(0.20))["opacities"].any()
    assert not track.get_gaussians_at(torch.tensor(0.30))["velocities"].any()
    loss = midpoint["means"].square().sum() + midpoint["quats"][..., 1:].square().sum()
    loss.backward()
    for param in (track.translation, track.root, track.body, track.means):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all() and param.grad.norm() > 0
    assert not track.body.grad[2].any()
    frozen = person(train_pose=False)
    assert not any(getattr(frozen, name).requires_grad for name in ("translation", "root", "body"))
    frozen.load_state_dict(track.state_dict())
    assert torch.equal(frozen.get_gaussians_at(torch.tensor(0.05))["means"], midpoint["means"])


def test_pelvis_pivot_in_world_to_scene_conversion():
    scene_from_world = torch.tensor([[0., -1., 0., 1.], [1., 0., 0., 2.],
                                     [0., 0., 1., 3.], [0., 0., 0., 1.]])
    t = torch.tensor(0.05)
    mesh_world, _ = person().mesh_at(t)
    mesh_scene, _ = person(transform=scene_from_world).mesh_at(t)
    assert torch.allclose(mesh_scene, mesh_world @ scene_from_world[:3, :3].T + scene_from_world[:3, 3], atol=1e-6)


def test_rotation_near_half_turn_and_antipodal_sign():
    q = torch.tensor([0., 0., 1., 0.], requires_grad=True)
    rot = quat_matrix(q)
    assert torch.allclose(quat_matrix(matrix_quat(rot)), rot)
    interpolated = slerp(q, -q, torch.tensor(0.5))
    assert torch.allclose(quat_matrix(interpolated), rot)
    interpolated.sum().backward()
    assert torch.isfinite(q.grad).all()


def test_crop_and_hierarchical_paired_metrics():
    assert crop_bounds([0, 0, 9, 10], 1080, 1920) == (0, 0, 64, 64)
    assert crop_bounds([1900, 1000, 1920, 1080], 1080, 1920) == (1856, 1000, 1920, 1080)
    a = [{"scene_id": "011", "uuid": uuid, "camera": "front", "frame": frame, "scan": frame,
          "smpl_valid": True, "bin": "near", "psnr_db": score, "lidar_hits": 3}
         for uuid, frame, score in (("1", 0, 10), ("1", 1, 14), ("2", 0, 30))]
    b = [{**row, "psnr_db": row["psnr_db"] + 2} for row in a]
    summary = paired(a, b, "psnr_db", "near")
    assert summary["a"] == 21 and summary["delta_b_minus_a"] == 2
    assert summary["n_uuid"] == 2 and summary["n_crop"] == 3
    assert paired(a, b, "psnr_db", "far")["a"] is None


def test_replay_interpolates_sensor_pose_and_time():
    @dataclass
    class Camera:
        camera_to_worlds: torch.Tensor
        times: torch.Tensor
        metadata: dict

    a = Camera(torch.eye(4)[:3][None], torch.tensor([[0.0]]),
               {"cam_idx": 1, "sensor_idxs": torch.tensor([[2]]),
                "linear_velocities_local": torch.zeros(1, 3)})
    pose_b = torch.tensor([[0., -1., 0., 1.2], [1., 0., 0., 0.], [0., 0., 1., 0.]])
    b = Camera(pose_b[None], torch.tensor([[0.1]]),
               {"cam_idx": 2, "sensor_idxs": torch.tensor([[2]]),
                "linear_velocities_local": torch.tensor([[1., 0., 0.]])})
    camera = camera_between(a, b, 0.05, torch.tensor(0.5))
    assert torch.allclose(camera.times, torch.tensor([[0.05]]))
    assert torch.allclose(camera.camera_to_worlds[..., 3], torch.tensor([[0.6, 0., 0.]]))
    rotated = camera.camera_to_worlds[0, :3, :3] @ torch.tensor([1., 0., 0.])
    assert torch.allclose(rotated, torch.tensor([2**-0.5, 2**-0.5, 0.]), atol=1e-6)
    assert "cam_idx" not in camera.metadata and a.metadata["cam_idx"] == 1
    assert torch.equal(camera.metadata["sensor_idxs"], torch.tensor([[2]]))


def test_smooth_replay_camera_keeps_recorded_poses_and_continuous_velocity():
    times = [0.0, 0.1, 0.22, 0.3]
    angles = torch.tensor([0.0, 0.1, 0.6, 0.72])
    zero = torch.zeros_like(angles)
    rotation = quat_matrix(torch.stack((angles.cos(), zero, zero, angles.sin()), -1))
    position = torch.tensor([[0., 0., 0.], [0.07, 0., 0.], [0.41, 0., 0.], [0.6, 0., 0.]])
    recorded = torch.cat((rotation, position.unsqueeze(-1)), -1).unsqueeze(1)
    views = [SimpleNamespace(camera_to_worlds=pose) for pose in recorded]
    sample_times = [-0.04, *times, 0.35, 0.1 - 1e-4, 0.1 + 1e-4]
    samples = smooth_camera_poses(views, times, sample_times)
    assert torch.allclose(samples[1:5], recorded, atol=1e-6)
    assert torch.allclose(samples[0], recorded[0], atol=1e-6)
    assert torch.allclose(samples[5], recorded[-1], atol=1e-6)
    before, after = (samples[2] - samples[6]) / 1e-4, (samples[7] - samples[2]) / 1e-4
    assert torch.allclose(before, after, atol=0.04)


def test_smooth_replay_smpl_root_keeps_keyframes_and_does_not_bridge_gaps():
    track = person(frame_ids=(0, 1, 2, 4, 5))
    with torch.no_grad():
        track.translation[:, 0] = torch.tensor([0.0, 0.08, 0.3, 0.8, 1.0])
    initial = [track.get_gaussians_at(time)["means"] for time in track.times]
    keys = track.state_dict().keys()
    track.prepare_smooth_replay()
    assert track.state_dict().keys() == keys
    for time, means in zip(track.times, initial, strict=True):
        assert torch.allclose(track.get_gaussians_at(time)["means"], means)
    first = track.get_gaussians_at(track.times[1] - 1e-4)["velocities"]
    second = track.get_gaussians_at(track.times[1] + 1e-4)["velocities"]
    assert torch.allclose(first, second, atol=0.015)
    assert not track.get_gaussians_at(torch.tensor(0.3))["opacities"].any()
    assert not track.get_gaussians_at(torch.tensor(0.03), motion=False)["velocities"].any()
