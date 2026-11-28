"""Dataset seed for R3 deformable pedestrians on a tiny synthetic PandaSet scene."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from research.deform_seed import build_seed, load_seed, save_seed

FX, CX, CY, WIDTH, HEIGHT = 800.0, 160.0, 120.0, 320, 240
# Camera at the origin looking along world +x: camera x is world -y, camera y is world -z.
CAMERA_TO_WORLD = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
RED, BLUE = (220, 20, 20), (20, 40, 220)


def cuboid(uuid, label, center, stationary=False, sensor=-1):
    return {"uuid": uuid, "label": label, "stationary": stationary, "cuboids.sensor_id": sensor, "yaw": 0.0,
            **{f"position.{a}": v for a, v in zip("xyz", center)},
            **{f"dimensions.{a}": v for a, v in zip("xyz", (0.6, 0.6, 1.8))}}


def write_scene(root):
    scene = root / "030"
    for folder in ("lidar", "annotations/cuboids", "camera/front_camera"):
        (scene / folder).mkdir(parents=True)
    centers = [(10.0, 0.0, 0.0), (10.0, 0.5, 0.0)]
    (scene / "lidar" / "timestamps.json").write_text(json.dumps([0.0, 0.1]))
    camera = scene / "camera" / "front_camera"
    x, y, z, w = Rotation.from_matrix(CAMERA_TO_WORLD).as_quat()
    pose = {"position": {"x": 0, "y": 0, "z": 0}, "heading": {"w": w, "x": x, "y": y, "z": z}}
    (camera / "intrinsics.json").write_text(json.dumps({"fx": FX, "fy": FX, "cx": CX, "cy": CY}))
    (camera / "poses.json").write_text(json.dumps([pose, pose]))
    (camera / "timestamps.json").write_text(json.dumps([0.0, 0.1]))
    face_y, face_z = np.meshgrid(np.arange(-0.25, 0.26, 0.05), np.arange(-0.85, 0.86, 0.05))
    wall_y, wall_z = np.meshgrid(np.arange(-3, 3.01, 0.1), np.arange(-1.5, 1.51, 0.1))
    for frame, center in enumerate(centers):
        face = np.column_stack((np.full(face_y.size, center[0] - 0.3), face_y.ravel() + center[1], face_z.ravel()))
        wall = np.column_stack((np.full(wall_y.size, 20.0), wall_y.ravel(), wall_z.ravel()))
        points = np.vstack((face, wall, [[10.0, center[1], 0.0]]))
        sensor = np.r_[np.zeros(len(face) + len(wall)), 1]  # the PandarGT return is ignored
        pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "z": points[:, 2], "d": sensor}).to_pickle(
            scene / "lidar" / f"{frame:02d}.pkl.gz")
        pd.DataFrame([cuboid("walker", "Pedestrian", center), cuboid("idle", "Pedestrian", (12, -2, 0), True),
                      cuboid("car", "Car", (15, 3, 0)), cuboid("front-only", "Pedestrian", (12, 2, 0), sensor=1)]
                     ).to_pickle(scene / "annotations" / "cuboids" / f"{frame:02d}.pkl.gz")
        image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        image[:] = BLUE
        # Paint the person's front face, as the camera sees it.
        left, right = (CX - (center[1] + s * 0.3) / 9.7 * FX for s in (1, -1))
        top, bottom = (CY - s * 0.9 / 9.7 * FX for s in (1, -1))
        image[int(top):int(bottom) + 1, int(left):int(right) + 1] = RED
        Image.fromarray(image).save(camera / f"{frame:02d}.jpg", quality=100)
    return root


@pytest.fixture
def raw(tmp_path):
    return write_scene(tmp_path / "raw")


def test_seed_collects_lidar_in_the_box_frame_and_colours_it(raw):
    result, meta = build_seed(raw, "030", cameras=("front",), target=1000, max_points=5000)
    assert set(result) == {"walker"}  # stationary, non-pedestrian and front-LiDAR-only cuboids are skipped
    points, colours, source = result["walker"]
    lidar = source == 0
    assert lidar.sum() == 2 * 11 * 35 == meta["pedestrians"]["walker"]["lidar_points"]
    # Both frames land on the same box-frame face although the person moved 0.5 m.
    np.testing.assert_allclose(points[lidar, 0], -0.3, atol=1e-5)
    assert np.abs(points[lidar, 1]).max() <= 0.25 + 1e-5
    assert meta["pedestrians"]["walker"]["lidar_coloured"] == lidar.sum()
    reds = colours[lidar]
    assert (reds[:, 0] > 180).mean() > 0.95 and (reds[:, 2] < 80).mean() > 0.95


def test_seed_fills_up_to_the_target_inside_the_body_with_camera_colours(raw):
    result, meta = build_seed(raw, "030", cameras=("front",), target=1000, max_points=5000)
    points, colours, source = result["walker"]
    fill = source == 1
    assert fill.sum() == 1000 - 770 == meta["pedestrians"]["walker"]["fill_points"]
    inside = (points[fill, 0] / 0.3) ** 2 + (points[fill, 1] / 0.3) ** 2
    assert inside.max() <= 1 + 1e-5 and np.abs(points[fill, 2]).max() <= 0.9 + 1e-5
    assert (colours[fill, 0] > 180).mean() > 0.9 and (colours[fill, 2] < 80).mean() > 0.9


def test_seed_caps_lidar_and_round_trips(raw, tmp_path):
    result, meta = build_seed(raw, "030", cameras=("front",), target=300, max_points=500)
    points, _, source = result["walker"]
    assert len(points) == 500 and not source.any()  # LiDAR alone exceeds the target, so no fill
    path = tmp_path / "seed" / "deform-seed.npz"
    save_seed(path, result, meta)
    loaded = load_seed(path, "030")
    for expected, actual in zip(result["walker"], loaded["walker"]):
        np.testing.assert_array_equal(expected, actual)
    with pytest.raises(ValueError, match="another scene"):
        load_seed(path, "011")
