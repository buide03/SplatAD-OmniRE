"""ViTPose keypoints for R3 far walkers: boxes, grouping and the stored camera, with a stand-in estimator."""
import json

import numpy as np
import pytest

from research.keypoints import COCO, detect, expanded_box, load_keypoints, observations_by_image, scatter
from tests.test_deform_seed import write_scene


def test_boxes_grow_and_stay_inside_the_image():
    assert expanded_box([100, 200, 120, 260], 1920, 1080) == pytest.approx([97, 191, 26, 78])
    assert expanded_box([0, 0, 40, 100], 1920, 1080) == pytest.approx([0, 0, 46, 115])


def test_detect_stores_keypoints_with_their_camera(tmp_path):
    raw = write_scene(tmp_path / "raw")
    rows = [{"uuid": "walker", "camera": "front", "frame": f, "bbox_xyxy": [140, 40, 180, 200], "usable": True}
            for f in (0, 1)] + [{"uuid": "hidden", "camera": "front", "frame": 0, "bbox_xyxy": [0, 0, 5, 5],
                                 "usable": False}]
    manifest = {"scene_id": "030", "observations": rows}
    assert sorted(observations_by_image(manifest)) == [("front", 0), ("front", 1)]
    seen = []

    def estimator(image, boxes):
        seen.append((image.size, boxes))
        return [(np.tile([[150.0, 120.0]], (17, 1)), np.linspace(0, 1, 17)) for _ in boxes]

    found = detect(raw, "030", manifest, estimator)
    assert [r["frame"] for r in found] == [0, 1] and seen[0][0] == (320, 240)
    row = found[0]
    assert len(row["keypoints_xy"]) == len(row["scores"]) == len(COCO) == 17
    assert row["intrinsics"] == {"fx": 800.0, "fy": 800.0, "cx": 160.0, "cy": 120.0}
    np.testing.assert_allclose(row["camera_rotation"], [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=1e-6)
    assert row["timestamp_s"] == 0.0 and found[1]["timestamp_s"] == pytest.approx(0.1)

    path = tmp_path / "keypoints.json"
    path.write_text(json.dumps({"schema_version": 1, "scene_id": "030", "manifest_sha256": "m", "observations": found}))
    assert load_keypoints(path, "030", "m") == found
    with pytest.raises(ValueError, match="another manifest"):
        load_keypoints(path, "030", "other")


def test_scatter_places_labelled_keypoints_and_leaves_the_rest_unscored():
    xy, score = scatter({"keypoints": np.array([[1.0, 2.0], [3.0, 4.0]]), "scores": np.array([0.9, 0.5]),
                         "labels": np.array([15, 16])})
    assert xy.shape == (17, 2) and score.shape == (17,)
    np.testing.assert_array_equal(xy[15:], [[1, 2], [3, 4]])
    assert score[15] == 0.9 and score[:15].sum() == 0
