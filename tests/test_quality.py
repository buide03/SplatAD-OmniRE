from pathlib import Path
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
    dataset = SimpleNamespace(image_filenames=[
        Path("030/camera/front_left_camera/08.jpg"), "030/camera/front_camera/08.jpg",
    ])
    pipeline = SimpleNamespace(model=model, datamanager=SimpleNamespace(train_dataset=dataset), training=True)
    pipeline.eval = lambda: setattr(pipeline, "training", False)
    pipeline.train = lambda mode: setattr(pipeline, "training", mode)

    def sample_camera(manager, index):
        assert index == 1
        return None, target

    monkeypatch.setattr(evaluate, "camera_sample", sample_camera)
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


def test_reviewed_pedestrian_frames_preserve_global_sampling_and_use_tight_crops(tmp_path, monkeypatch):
    from PIL import Image

    scene = tmp_path / "028"
    camera_dir = scene / "camera/front_camera"
    camera_dir.mkdir(parents=True)
    dump_json(camera_dir / "timestamps.json", [0.0, 0.1, 0.2])
    for frame in range(3):
        Image.new("RGB", (64, 64)).save(camera_dir / f"{frame:02d}.jpg")
    review = {"scene_id": "028", "observations": [
        {"camera": "front", "frame": 1, "uuid": "person", "bbox_xyxy": [20, 20, 30, 36], "pose_monitor": True},
    ]}
    monkeypatch.setattr(quality, "camera_at", lambda *args: None)
    monkeypatch.setattr(quality, "box_records", lambda *args: [])
    examples = quality.freeze_examples(scene, ["front"], 2, review)
    assert [(image["frame"], image["person_only"]) for image in examples["images"]] == [(0, False), (1, True), (2, False)]
    target = torch.zeros(64, 64, 3)
    predicted = target.clone()
    predicted[20:36, 19:30] = 1
    model = SimpleNamespace(get_camera_outputs=lambda _: {"rgb": predicted, "background": None},
                            get_gt_img=lambda image: image, composite_with_background=lambda image, _: image,
                            ssim=lambda a, b: torch.tensor(0.5))
    pipeline = SimpleNamespace(model=model, datamanager=None, training=False, eval=lambda: None, train=lambda mode: None)
    monkeypatch.setattr(evaluate, "camera_indices", lambda manager: {("front", frame): frame for frame in range(3)})
    monkeypatch.setattr(evaluate, "camera_sample", lambda manager, index: (None, target))
    metrics = quality.measure_quality(pipeline, examples)
    assert metrics["scene"]["samples"] == 2 and metrics["vehicle"]["samples"] == 0
    assert metrics["person"] == {"samples": 1, "psnr_db": 0.0, "ssim": 0.5}
