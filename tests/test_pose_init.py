from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from research.geometry import JOINTS
from research.pose_init import merge_initialization


IDENTITY = [1, 0, 0, 0]


def observation(index, valid=True, offset=0, displacement=0):
    return {"frame_idx": index, "timestamp_s": 100 + index / 10 + offset, "valid": valid,
            "source_cam": "front" if valid else None, "pose_source": "4d_humans" if valid else None,
            "root_translation": [displacement, 0, 0] if valid else None,
            "root_quat_wxyz": IDENTITY if valid else None,
            "body_quats_wxyz": [IDENTITY] * 23 if valid else None}


def document(uuid, *frames):
    return {"schema_version": 1, "scene_id": "011", "coordinate_system": "pandaset_world",
            "quaternion_order": "wxyz", "length_unit": "m", "time_unit": "s",
            "time_reference": "pandaset_camera_epoch", "joint_order": list(JOINTS),
            "translation_definition": "cuboid_center_minus_posed_mesh_bbox_center",
            "smpl_version": "smpl_neutral_v1.1", "body_sha256": "body",
            "tracks": [{"uuid": uuid, "betas": [0] * 10, "frames": list(frames)}]}


def test_reference_report_and_fill_keep_valid_local_pose_and_uuid():
    primary = document("same", observation(0, displacement=1), observation(1, False), observation(2))
    reference = document("same", observation(0, displacement=2), observation(1, offset=0.004))
    reference["tracks"].extend(document("other", observation(0))["tracks"])
    snapshot = deepcopy(primary)
    compared, comparison = merge_initialization(primary, reference, "compare")
    assert not compared["tracks"][0]["frames"][1]["valid"]
    assert comparison["counts"]["available_fills"] == 1
    assert comparison["counts"]["filled_frames"] == 0
    assert comparison["median_difference"]["root_translation_m"] == 1
    final, report = merge_initialization(primary, reference, "fill")
    assert primary == snapshot
    assert [track["uuid"] for track in final["tracks"]] == ["same"]
    local, filled, _ = final["tracks"][0]["frames"]
    assert local["root_translation"] == [1, 0, 0] and local["pose_origin"] == "local_4d_humans"
    assert filled["valid"] and filled["pose_origin"] == "omnire_reference"
    assert report["filled_frames"] == [{"uuid": "same", "frame": 1, "camera": "front"}]


def test_fill_rejects_out_of_order_time_and_incompatible_reference():
    primary = document("same", observation(0), observation(1, False), observation(2))
    reference = document("same", observation(1, offset=0.2))
    final, report = merge_initialization(primary, reference, "fill")
    assert not final["tracks"][0]["frames"][1]["valid"]
    assert report["counts"]["timestamp_conflicts"] == 1
    reference["smpl_version"] = "smpl_neutral_v1.0"
    with pytest.raises(ValueError, match="same SMPL body"):
        merge_initialization(primary, reference, "fill")
    direct, empty = merge_initialization(primary, None, "fill")
    assert not direct["tracks"][0]["frames"][1]["valid"] and not empty["reference_available"]


def test_fill_recenters_reference_pose_to_keep_local_shape():
    primary = document("same", observation(0), observation(1, False))
    reference = document("same", observation(1, displacement=3))
    reference["tracks"][0]["betas"][0] = 2
    body = SimpleNamespace(parents=(-1,) + (0,) * 23,
                           weights=torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), 24).float())

    def canonical(betas):
        vertices = torch.tensor([[0., 0., 0.], [2., 0., 0.]]) + betas[0] * torch.tensor([1., 0., 0.])
        return vertices, torch.zeros((24, 3)), torch.eye(4).repeat(24, 1, 1)

    body.canonical = canonical
    final, _ = merge_initialization(primary, reference, "fill", body)
    assert final["tracks"][0]["betas"] == primary["tracks"][0]["betas"]
    assert final["tracks"][0]["frames"][1]["root_translation"] == [5, 0, 0]
