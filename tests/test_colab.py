import json
import subprocess
import sys

import pytest
import torch

from research.colab import ColabProgress, checkpoint_step, latest_checkpoint, retry_environment


def test_resume_ignores_unfinished_checkpoints(tmp_path):
    (tmp_path / "config.yml").touch()
    initial = tmp_path / "initial.ckpt"
    torch.save({"step": -1}, initial)
    folder = tmp_path / "nerfstudio_models"
    folder.mkdir()
    for step in (8, 10):
        torch.save({"step": step}, folder / f"step-{step}.ckpt")
    (folder / "step-14.ckpt").write_bytes(b"PK\x03\x04")
    torch.save({"step": 20}, folder / "step-20.partial")
    assert checkpoint_step(latest_checkpoint(tmp_path)) == 11

    (folder / "step-10.ckpt").unlink()
    (folder / "step-8.ckpt").unlink()
    assert latest_checkpoint(tmp_path) is None
    assert latest_checkpoint(tmp_path, include_initial=True) == initial
    assert checkpoint_step(initial) == 0


def test_retries_only_recognized_environment_errors():
    events = []

    def open_environment():
        events.append("launch")
        if len(events) == 1:
            raise subprocess.CalledProcessError(1, "python", output="ImportError: undefined symbol")
        return "resumed"

    assert retry_environment(open_environment, lambda: events.append("repair")) == "resumed"
    assert events == ["launch", "repair", "launch"]

    def missing_data():
        raise subprocess.CalledProcessError(1, "python", output="FileNotFoundError: missing PandaSet")

    with pytest.raises(subprocess.CalledProcessError):
        retry_environment(missing_data, lambda: pytest.fail("Rebuilding packages cannot restore PandaSet"))


def test_phase_logs_survive_a_restart(tmp_path):
    status = ColabProgress(tmp_path)
    with pytest.raises(subprocess.CalledProcessError), status.phase("train-028-R2"):
        status.run([sys.executable, "-c", "print('First attempt'); raise SystemExit(1)"], tmp_path)

    state = json.loads(status.path.read_text(encoding="utf-8"))
    assert state["train-028-R2"]["status"] == "failed"
    assert "First attempt" in state["train-028-R2"]["error"]

    restarted = ColabProgress(tmp_path)
    with restarted.phase("train-028-R2"):
        restarted.run([sys.executable, "-c", "print('Second attempt')"], tmp_path)
    record = json.loads(restarted.path.read_text(encoding="utf-8"))["train-028-R2"]
    assert (record["status"], record["attempt"]) == ("complete", 2)
    log = (tmp_path / record["log"]).read_text(encoding="utf-8")
    assert "First attempt" in log and "Second attempt" in log
