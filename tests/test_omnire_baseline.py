"""CPU checks for the OmniRe baseline settings, shared RGB scoring and the four-model summary."""
import glob
import sys
from types import ModuleType

import pytest
import torch

from research.omnire_baseline import OMNIRE_CAMERAS, checkpoints, read_gzipped_pandaset, shared_settings
from research.quality import score_images
from research.summary import markdown, summarize


def test_omnire_sees_the_splatad_data_and_budget():
    settings = dict(item.split("=", 1) for item in shared_settings("/data/processed", "030", 10000, 42, 2500))
    assert settings["dataset"] == "pandaset/6cams"
    assert settings["data.scene_idx"] == "30"
    assert settings["data.pixel_source.downscale_when_loading"] == "[1,1,1,1,1,1]"
    assert settings["data.pixel_source.test_image_stride"] == "0"
    assert settings["data.lidar_source.only_use_360_lidar"] == "True"
    assert settings["trainer.optim.num_iters"] == "10000" and settings["seed"] == "42"
    # DriveStudio PandaSet camera ids 0-5.
    assert OMNIRE_CAMERAS.index("front") == 0 and OMNIRE_CAMERAS.index("back") == 5


def test_checkpoints_are_ordered_and_final_counts_as_the_budget(tmp_path):
    for name in ("checkpoint_05000.pth", "checkpoint_final.pth", "checkpoint_02500.pth", "config.yaml"):
        (tmp_path / name).touch()
    assert [step for step, _ in checkpoints(tmp_path, 10000)] == [2500, 5000, 10000]


def test_staged_lidar_and_cuboids_are_found(tmp_path, monkeypatch):
    class Sensor:
        """The pinned devkit's lookup: sweeps are the files matching *.<_data_file_extension>."""
        _data_file_extension = property(lambda self: "pkl")

        def files(self, folder):
            return sorted(glob.glob(f"{folder}/*.{self._data_file_extension}"))

    modules = {"pandaset": ModuleType("pandaset"), "pandaset.annotations": ModuleType("pandaset.annotations"),
               "pandaset.sensors": ModuleType("pandaset.sensors")}
    modules["pandaset.annotations"].Cuboids = type("Cuboids", (Sensor,), {})
    modules["pandaset.sensors"].Lidar = type("Lidar", (Sensor,), {})
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for frame in range(2):
        (tmp_path / f"{frame:02d}.pkl.gz").touch()
    lidar, cuboids = modules["pandaset.sensors"].Lidar(), modules["pandaset.annotations"].Cuboids()
    assert lidar.files(tmp_path) == []  # The failure on Colab: an empty sweep list.
    read_gzipped_pandaset()
    assert len(lidar.files(tmp_path)) == len(cuboids.files(tmp_path)) == 2


def similarity(a, b):
    # Stand-in for pytorch_msssim.SSIM, which only the Colab environments install.
    return 1 - (a - b).square().mean()


def test_shared_scoring_crops_people_and_skips_person_only_scene_scores():
    examples = {"schema": 2, "images": [
        {"camera": "back", "frame": 0, "height": 40, "width": 64, "person_only": False,
         "crops": [{"scope": "person", "bbox": [10, 10, 14, 30]}, {"scope": "vehicle", "bbox": [0, 0, 64, 40]}]},
        {"camera": "front", "frame": 5, "height": 40, "width": 64, "person_only": True,
         "crops": [{"scope": "person", "bbox": [30, 5, 40, 35]}]},
    ]}
    torch.manual_seed(0)
    target = torch.rand(40, 64, 3)
    noisy = (target + 0.1 * torch.randn_like(target)).clamp(0, 1)
    perfect = score_images(examples, lambda sample: (target, target), similarity)
    assert {scope: perfect[scope]["samples"] for scope in perfect} == {"scene": 1, "vehicle": 1, "person": 2}
    assert perfect["scene"]["psnr_db"] == pytest.approx(120)
    worse = score_images(examples, lambda sample: (noisy, target), similarity)
    assert worse["person"]["psnr_db"] < 30 and worse["scene"]["ssim"] < perfect["scene"]["ssim"]
    with pytest.raises(ValueError, match="size changed"):
        score_images(examples, lambda sample: (target[:20], target[:20]), similarity)


def report(variant, steps, offset, digest="samples"):
    scopes = {scope: {"samples": 4, "psnr_db": 20.0 + offset, "ssim": 0.5 + offset / 100}
              for scope in ("scene", "vehicle", "person")}
    return {"scene": "030", "variant": variant, "run_id": "run", "quality_sha256": digest,
            "steps": [{"step": step, "scopes": scopes} for step in steps]}


def test_summary_keeps_shared_steps_and_deltas_against_splatad():
    grid = list(range(0, 10001, 250))
    summary = summarize([report("OmniRe", [2500, 5000, 7500, 10000], 1.0), report("R3", grid, 2.0),
                         report("SplatAD", grid, 0.0), report("R2", grid, -0.5)])
    assert summary["variants"] == ["SplatAD", "R2", "R3", "OmniRe"]
    assert summary["steps"] == [2500, 5000, 7500, 10000]
    row = next(r for r in summary["rows"] if r["step"] == 10000 and r["scope"] == "person" and r["variant"] == "R3")
    assert row["delta_psnr_db"] == pytest.approx(2.0)
    assert "22.00 dB (+2.00)" in markdown(summary)
    with pytest.raises(ValueError, match="frozen RGB samples"):
        summarize([report("SplatAD", grid, 0), report("OmniRe", grid, 0, digest="other")])


def test_omnire_video_path_passes_through_recorded_camera_poses():
    import numpy as np
    from scipy.spatial.transform import Rotation

    from research.omnire_baseline import camera_path

    times = [0.0, 0.1, 0.2, 0.3]
    c2w = torch.eye(4).repeat(4, 1, 1)
    for i in range(4):
        c2w[i, :3, :3] = torch.from_numpy(Rotation.from_euler("z", 5 * i, degrees=True).as_matrix())
        c2w[i, 0, 3] = 1.5 * i
    poses = camera_path(c2w, times, [0.0, 0.05, 0.1, 0.3])
    assert np.allclose(poses[[0, 2, 3]], c2w[[0, 1, 3]].double().numpy(), atol=1e-6)
    assert 0 < poses[1, 0, 3] < 1.5


def test_run_record_names_the_native_splatad_baseline(tmp_path):
    from types import SimpleNamespace

    from research.evaluate import run_record

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    config = SimpleNamespace(
        pipeline=SimpleNamespace(model=SimpleNamespace(), datamanager=SimpleNamespace(
            dataparser=SimpleNamespace(sequence="030"))),
        timestamp="run", get_base_dir=lambda: tmp_path)
    record = run_record(config, tmp_path / "step-000009999.ckpt", 9999, manifest, {"observations": []}, "checkpoint")
    assert (record["scene"], record["variant"], record["step"]) == ("030", "SplatAD", 9999)


def test_omnire_log_uses_the_r2_r3_gradient_names(tmp_path):
    import json
    from types import SimpleNamespace

    from research.omnire_baseline import gradient_row, truncate_jsonl

    translation = torch.nn.Parameter(torch.zeros(3, 1, 3))
    network = torch.nn.Parameter(torch.zeros(4))
    frozen = torch.nn.Parameter(torch.zeros(2))
    translation.grad = torch.full_like(translation, 0.5)
    network.grad = torch.ones_like(network)
    trainer = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[
        {"name": "SMPLNodes#ins_translation", "params": [translation]},
        {"name": "DeformableNodes#deform_network", "params": [network]},
        {"name": "Background#xyz", "params": [frozen]},
    ]))
    row = gradient_row(trainer, 50, {"rgb_loss": torch.tensor(0.2), "smpl_temporal_smooth": torch.tensor(0.01)})
    assert row["scene_loss"] == pytest.approx(0.21)
    assert row["gradients"]["smpl_translation"]["norm"] == pytest.approx(1.5)
    assert row["gradients"]["deformable_network"]["norm"] == pytest.approx(2.0)
    assert row["gradients"]["background_means"]["state"] == "none"
    assert row["gradients"]["smpl_body_rotation"]["state"] == "none"

    log = tmp_path / "training.jsonl"
    log.write_text("".join(json.dumps({"step": s}) + "\n" for s in (0, 25, 1000, 1025)) + "torn line")
    truncate_jsonl(log, 1000)
    assert [json.loads(line)["step"] for line in log.read_text().splitlines()] == [0, 25]


def test_omnire_resume_state_round_trips_model_optimizer_and_rng(tmp_path):
    import random
    from types import SimpleNamespace

    from research.omnire_baseline import save_resume_state

    weight = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.Adam([weight], lr=0.1)
    weight.grad = torch.ones(3)
    optimizer.step()
    trainer = SimpleNamespace(optimizer=optimizer, state_dict=lambda only_model=True: {"models": {"w": weight.detach()}})
    random.seed(9)
    torch.manual_seed(9)
    path = tmp_path / "resume.pth"
    save_resume_state(path, trainer, 1999)
    expected = (random.random(), torch.rand(1))
    state = torch.load(path, weights_only=False)
    assert state["step"] == 1999 and not path.with_suffix(".partial").exists()
    assert torch.equal(state["model"]["models"]["w"], weight.detach())
    assert state["optimizer"]["state"][0]["step"] == 1
    random.setstate(state["rng"]["python"])
    torch.set_rng_state(state["rng"]["torch"])
    assert (random.random(), torch.rand(1)) == expected
