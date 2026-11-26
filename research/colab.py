"""Persist Colab phase logs and select completed PyTorch checkpoints."""
import json
import os
import re
import signal
import subprocess
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile


ENV_ERROR = re.compile(
    r"ModuleNotFoundError|ImportError|PackageNotFoundError|undefined symbol|"
    r"cannot open shared object file|"
    r"numpy\.dtype size changed|operator torchvision::\w+ does not exist|"
    r"Error building extension ['\x22][^'\x22]*gsplat",
    re.IGNORECASE,
)
CHECKPOINT_NAME = re.compile(r"step-(\d+)\.ckpt")


def checkpoint_step(path):
    """Return the number of completed steps represented by a checkpoint."""
    return int(CHECKPOINT_NAME.fullmatch(path.name)[1]) + 1 if path and path.name != "initial.ckpt" else 0


def complete_checkpoint(path):
    try:
        with ZipFile(path) as archive:
            return any(name.endswith("/data.pkl") for name in archive.namelist())
    except (FileNotFoundError, IsADirectoryError, BadZipFile):
        return False


def latest_checkpoint(folder, include_initial=False):
    """Ignore unfinished writes and reject truncated ZIP checkpoints without loading tensors."""
    if not (folder / "config.yml").is_file():
        return None
    candidates = sorted(
        (path for path in (folder / "nerfstudio_models").glob("step-*.ckpt") if CHECKPOINT_NAME.fullmatch(path.name)),
        key=checkpoint_step, reverse=True,
    )
    if include_initial:
        candidates.append(folder / "initial.ckpt")
    for path in candidates:
        if complete_checkpoint(path):
            return path
        if path.exists():
            print(f"Skipping incomplete checkpoint: {path}", flush=True)
    return None


def retry_environment(action, repair):
    try:
        return action()
    except subprocess.CalledProcessError as error:
        if not ENV_ERROR.search(error.output or ""):
            raise
        print("Dependency error. Rebuilding the isolated environment before one retry.", flush=True)
        repair()
    return action()


class ColabProgress:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "progress.json"
        self.records = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        self.active = None

    def _record(self, phase, status, **details):
        timestamp = datetime.now(timezone.utc).isoformat()
        self.records[phase].update(status=status, updated_at=timestamp, **details)
        temporary = self.path.with_suffix(".partial")
        temporary.write_text(json.dumps(self.records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        with (self.folder / self.records[phase]["log"]).open("a", encoding="utf-8") as log:
            print(f"[{timestamp}] {status}", file=log)
            if details.get("error"):
                print(details["error"], file=log)

    @contextmanager
    def phase(self, phase):
        previous = self.active
        self.active = phase
        self.records[phase] = {"attempt": self.records.get(phase, {}).get("attempt", 0) + 1, "log": f"{phase}.log"}
        self._record(phase, "running")
        try:
            yield
        except KeyboardInterrupt:
            self._record(phase, "interrupted")
            raise
        except subprocess.CalledProcessError as error:
            status = "interrupted" if error.returncode in (-signal.SIGINT, -signal.SIGTERM, 130, 143) else "failed"
            self._record(phase, status, error=(error.output or str(error))[-1500:])
            raise
        except Exception as error:
            self._record(phase, "failed", error=str(error))
            raise
        else:
            self._record(phase, "complete")
        finally:
            self.active = previous

    def run(self, args, cwd):
        args = [str(arg) for arg in args]
        tail = deque(maxlen=100)
        logfile = self.folder / f"{self.active or 'commands'}.log"
        with logfile.open("a", encoding="utf-8", buffering=1) as log:
            print(f"\n[{datetime.now(timezone.utc).isoformat()}] {subprocess.list2cmdline(args)}", file=log)
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            with subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  encoding="utf-8", errors="replace") as process:
                try:
                    for line in process.stdout:
                        tail.append(line)
                        log.write(line)
                        print(line, end="", flush=True)
                    returncode = process.wait()
                except KeyboardInterrupt:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT if os.name == "posix" else signal.SIGTERM)
                        try:
                            process.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    raise
            if returncode:
                raise subprocess.CalledProcessError(returncode, args, output="".join(tail))
