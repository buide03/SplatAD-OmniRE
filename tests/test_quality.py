from types import SimpleNamespace

import pytest
import torch

from research import evaluate, quality
from research.prepare import dump_json, read_json


def test_quality_measures_full_image_and_separate_vehicle_and_person_crops(monkeypatch):
    target = torch.zeros((64, 128, 3))
    prediction = target.clone()
    prediction[:, :64] = 0.5
    model = SimpleNamespace(info={"training": True}, last_size=(16, 32))
    model.get_gt_img = lambda image: image
    model.composite_with_background = lambda image, background: image
    ssim_shapes = []

    def ssim(a, b):
        ssim_shapes.append(a.shape)
        return torch.tensor(1.0 if torch.equal(a, b) else 0.5)

    def render(camera):
        assert not pipeline.training
        model.info = {"measurement": True}
        model.last_size = (64, 128)
        return {"rgb": prediction, "background": torch.zeros(3)}

    model.ssim = ssim
    model.get_camera_outputs = render
    pipeline = SimpleNamespace(model=model, datamanager=None, training=True)
    pipeline.eval = lambda: setattr(pipeline, "training", False)
    pipeline.train = lambda mode: setattr(pipeline, "training", mode)
    monkeypatch.setattr(evaluate, "camera_indices", lambda manager: {("front", 8): 0})
    monkeypatch.setattr(evaluate, "camera_sample", lambda manager, index: (None, target))
    examples = {"images": [{"camera": "front", "frame": 8, "width": 128, "height": 64, "crops": [
        {"scope": "vehicle", "bbox": [0, 0, 64, 64]}, {"scope": "person", "bbox": [64, 0, 128, 64]},
    ]}]}
    stats = quality.measure_quality(pipeline, examples)
    assert stats["scene"]["psnr_db"] == pytest.approx(9.0309)
    assert stats["vehicle"]["psnr_db"] == pytest.approx(6.0206)
    assert stats["person"] == {"samples": 1, "psnr_db": 120.0, "ssim": 1.0}
    assert stats["vehicle"]["ssim"] == 0.5
    assert ssim_shapes == [(1, 3, 64, 128), (1, 3, 64, 64), (1, 3, 64, 64)]
    assert pipeline.training and model.info == {"training": True} and model.last_size == (16, 32)


def test_quality_resume_retains_initial_values_and_discards_uncheckpointed_steps(tmp_path, monkeypatch):
    sample = tmp_path / "probe.json"
    output = tmp_path / "report"
    dump_json(sample, {"schema": 1, "scene_id": "011", "images": [{"frame": 0}]})
    monkeypatch.setattr(quality, "measure_quality", lambda pipeline, examples: {"scene": {"samples": 1}})
    history = quality.QualityHistory(sample, output, "011", "R2", "run", 0)
    for step in (0, 50, 100, 150):
        history.capture(None, step)
    resumed = quality.QualityHistory(sample, output, "011", "R2", "run", 100)
    resumed.capture(None, 100)
    assert [row["step"] for row in read_json(output / "quality.json")["steps"]] == [0, 50, 100]
    assert read_json(output / "run.json")["step"] == 100
    dump_json(sample, {"schema": 1, "scene_id": "011", "images": [{"frame": 1}]})
    with pytest.raises(ValueError, match="another run or sample set"):
        quality.QualityHistory(sample, output, "011", "R2", "run", 100)
