"""Train SplatAD and the pedestrian research variants on one Colab GPU."""
import argparse
import copy
import json
import os
import signal
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path
from time import perf_counter
from typing import Type

import torch
import yaml

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.models.splatad import SplatADModelConfig
from nerfstudio.scripts.train import main as launch

from .colab import complete_checkpoint
from .deformable import GAUSSIAN_PARAMETERS
from .evaluate import camera_indices
from .model import POSE_GROUPS, SMPL_VARIANTS, TRAINABLE_POSE_VARIANTS, PedestrianModelConfig
from .pose_monitor import POSE_ATTRS, PoseMonitor, drift, finite_number, gradient_stats
from .prepare import dump_json, read_json, sha256
from .quality import QualityHistory


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


@dataclass
class ResearchConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: ResearchTrainer)
    monitor_interval: int = 250
    gradient_interval: int = 25
    pose_monitor_interval: int = 50
    pose_monitor_dir: Path | None = None
    quality_set: Path | None = None
    quality_dir: Path | None = None
    quality_interval: int = 250

    def save_config(self):
        path = self.get_base_dir() / "config.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        temporary.write_text(yaml.dump(self), encoding="utf-8")
        temporary.replace(path)
        print(f"Saved config to {path}", flush=True)


class ResearchTrainer(Trainer):
    def train(self):
        self.stop_signal = None

        def request_stop(signum, _frame):
            self.stop_signal = signum

        signals = (signal.SIGINT, signal.SIGTERM)
        previous = {signum: signal.signal(signum, request_stop) for signum in signals}
        try:
            super().train()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        log = self.base_dir / "training.jsonl"
        if log.exists():
            preserved = []
            for line in log.read_text(encoding="utf-8").splitlines(keepends=True):
                try:
                    if json.loads(line)["step"] < self._start_step:
                        preserved.append(line)
                except json.JSONDecodeError:
                    continue
            temporary = log.with_suffix(".tmp")
            temporary.write_text("".join(preserved), encoding="utf-8")
            temporary.replace(log)
        self.sensor_counts = Counter()
        self.last_logged = {}
        self.monitor_seconds = 0.0
        self.run_start = perf_counter()
        model = self.pipeline.model
        self.native_baseline = type(model.config) is SplatADModelConfig
        if not self.native_baseline:
            dump_json(self.base_dir / "scene_routes.json", {
                "background_seeds": model.background_seeds, "background_gaussians": model.background_gaussians,
                "actors": model.actor_routes,
            })
            print(f"Background: {model.background_seeds:,} LiDAR seeds and {model.background_gaussians:,} Gaussians")
            for row in model.actor_routes:
                print(row)
        if getattr(model.config, "variant", None) == "R3" and model.config.person_rgb_weight:
            indices = camera_indices(self.pipeline.datamanager)
            constrained = {row["uuid"] for row in model.actor_routes if row["representation"] in ("smpl", "deformable")}
            for row in read_json(model.config.manifest_path)["observations"]:
                idx = indices.get((row["camera"], row["frame"]))
                if idx is not None and row["usable"] and row["uuid"] in constrained:
                    model.person_regions.setdefault(idx, []).append(tuple(map(int, row["bbox_xyxy"])))
        dump_json(self.base_dir / "environment.json", {
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "seed": self.config.machine.seed,
            "source_commits": read_json(Path(__file__).resolve().parents[1] / "versions.json"),
        })
        stats = self.base_dir / "run_stats.json"
        self.previous = read_json(stats) if self._start_step and stats.exists() else {}
        # A stats file may be newer than the checkpoint after an interruption.
        if self.previous and self.previous["step"] != self._start_step - 1:
            self.previous = {}
        self.quality = None
        if self.config.quality_set and self.config.quality_interval:
            self.quality = QualityHistory(
                self.config.quality_set, self.config.quality_dir or self.base_dir / "quality",
                self.config.pipeline.datamanager.dataparser.sequence, getattr(model.config, "variant", "SplatAD"),
                self.config.timestamp, self._start_step,
            )
        self.pose_monitor = None
        if self.config.pose_monitor_interval and not self.native_baseline and model.people:
            monitor = PoseMonitor(
                self.pipeline, read_json(model.config.manifest_path),
                self.config.pose_monitor_dir or self.base_dir / "pose_monitor",
                self.config.timestamp, self._start_step, model.config.manifest_digest,
            )
            if monitor.observations:
                self.pose_monitor = monitor
        self.callbacks.extend([
            TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.initial_monitor),
            TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN_ITERATION], self.periodic_monitor),
            TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN], self.final_monitor),
        ])

    def train_iteration(self, step):
        if self.native_baseline:
            return super().train_iteration(step)
        model = self.pipeline.model
        sample = min(self.last_logged.get(sensor, -self.config.gradient_interval)
                     for sensor in ("rgb", "lidar")) <= step - self.config.gradient_interval
        watched = self.pose_monitor.watched_uuids if self.pose_monitor else ()
        with torch.no_grad():
            before = {node.uuid: {name: getattr(node, name).detach().clone() for name in POSE_ATTRS}
                      for node in model.people if sample or node.uuid in watched}
        loss, losses, metrics = super().train_iteration(step)
        sensor = model.last_sensor
        self.sensor_counts[sensor] += 1
        due = step - self.last_logged.get(sensor, -self.config.gradient_interval) >= self.config.gradient_interval
        fixed_rgb = self.pose_monitor and model.last_rgb_index in self.pose_monitor.watched_indices
        if due:
            self.last_logged[sensor] = step
        if due or fixed_rgb:
            groups = model.get_param_groups()
            row = {"step": step, "sensor": sensor, "scene_loss": finite_number(float(loss.detach())),
                   "image_index": model.last_rgb_index,
                   "loss_terms": {name: finite_number(float(value.detach())) for name, value in losses.items()}}
            row["gradients"] = {tag: gradient_stats(groups.get(tag, [])) for tag in POSE_GROUPS}
            row["gradients"]["smpl_gaussians"] = gradient_stats(
                [p for tag, group in groups.items() if tag.startswith("smpl_") and tag not in POSE_GROUPS for p in group]
            )
            row["pose"] = {}
            with torch.no_grad():
                for node in model.people:
                    if node.uuid not in before:
                        continue
                    previous = before[node.uuid]
                    lo, hi, _, visible = node._span(model.last_actor_time.to(node.times).reshape(()))
                    if not bool(visible):
                        continue
                    indices = torch.stack((lo, hi)).unique()
                    row["pose"][node.uuid] = {
                        name: {
                            "gradient": gradient_stats([getattr(node, name)], indices),
                            "step_delta": drift(getattr(node, name)[indices], previous[name][indices], name != "translation"),
                            "initial_delta": drift(
                                getattr(node, name)[indices], getattr(node, f"initial_{name}")[indices],
                                name != "translation",
                            ),
                        }
                        for name in previous
                    }
            append_jsonl(self.base_dir / "training.jsonl", row)
            if fixed_rgb:
                self.pose_monitor.observe_training_step(row)
        return loss, losses, metrics

    def monitor(self, step):
        if self.native_baseline:
            return
        cfg = self.config
        elapsed_start = perf_counter()
        if cfg.pipeline.model.manifest_path is not None:
            from .evaluate import evaluate
            result = self.base_dir / "monitor" / f"step-{step:06d}"
            evaluate(self.pipeline, read_json(cfg.pipeline.model.manifest_path), result, probes=True)
        self.monitor_seconds += perf_counter() - elapsed_start

    def initial_monitor(self, step):
        if self.quality and step == self._start_step:
            if step == 0:
                self.save_initial_checkpoint()
            self.record_quality(step)
        if step == 0 and self.config.monitor_interval:
            self.monitor(0)
        if self.pose_monitor and step == self._start_step and not self.pose_monitor.steps:
            self.record_pose_monitor(step)

    def save_initial_checkpoint(self):
        path = self.base_dir / "initial.ckpt"
        temporary = path.with_suffix(".partial")
        pipeline = getattr(self.pipeline, "module", self.pipeline)
        torch.save({"step": -1, "pipeline": pipeline.state_dict(),
                    "optimizers": {key: opt.state_dict() for key, opt in self.optimizers.optimizers.items()},
                    "schedulers": {key: opt.state_dict() for key, opt in self.optimizers.schedulers.items()},
                    "scalers": self.grad_scaler.state_dict()}, temporary)
        temporary.replace(path)

    def record_quality(self, step):
        started = perf_counter()
        self.quality.capture(self.pipeline, step)
        self.monitor_seconds += perf_counter() - started

    def record_pose_monitor(self, step):
        if not self.pose_monitor or (self.pose_monitor.steps and self.pose_monitor.steps[-1]["step"] == step):
            return
        start = perf_counter()
        rates = []
        if self.pipeline.model.config.variant in TRAINABLE_POSE_VARIANTS:
            rates = [self.optimizers.optimizers[name].param_groups[0]["lr"] for name in POSE_GROUPS]
        self.pose_monitor.capture(step, rates)
        self.monitor_seconds += perf_counter() - start

    def write_stats(self, step, seconds):
        if self.native_baseline:
            return
        prior_counts = Counter(self.previous.get("sensor_batches", {}))
        prior_counts.update(self.sensor_counts)
        train_seconds = round(self.previous.get("train_wall_s", 0) + seconds, 3)
        dump_json(self.base_dir / "run_stats.json", {
            "step": step, "sensor_batches": dict(prior_counts),
            "train_wall_s": train_seconds, "step_per_s": sum(prior_counts.values()) / max(train_seconds, 1e-3),
            "peak_vram_mib": max(self.previous.get("peak_vram_mib", 0), torch.cuda.max_memory_allocated() / 2**20),
        })

    def periodic_monitor(self, step):
        if self.stop_signal is not None:
            self.save_checkpoint(step)
            print(f"Interrupted after {step + 1} steps. Checkpoint saved in {self.checkpoint_dir}.", flush=True)
            raise SystemExit(128 + self.stop_signal)
        completed = step + 1
        last = self._start_step + self.config.max_num_iterations
        if self.quality and (completed % self.config.quality_interval == 0 or completed == last):
            self.record_quality(step + 1)
        interval = self.config.monitor_interval
        if interval and (step + 1) % interval == 0:
            self.monitor(step + 1)
        if self.pose_monitor and (step + 1) % self.config.pose_monitor_interval == 0:
            self.record_pose_monitor(step + 1)
        if self.config.steps_per_save and (step % self.config.steps_per_save == 0):
            self.write_stats(step, perf_counter() - self.run_start - self.monitor_seconds)

    def final_monitor(self, step):
        if self.quality:
            self.record_quality(step + 1)
        if self.config.monitor_interval and (step + 1) % self.config.monitor_interval:
            self.monitor(step + 1)
        self.record_pose_monitor(step + 1)
        self.write_stats(step, perf_counter() - self.run_start - self.monitor_seconds)


def make_config(args):
    base = copy.deepcopy(method_configs["splatad"])
    settings = {f.name: getattr(base, f.name) for f in fields(TrainerConfig) if f.name != "_target"}
    config = ResearchConfig(**settings)
    config.method_name = "splatad-smpl"
    config.experiment_name = f"{args.scene}-{args.variant}"
    config.output_dir = args.output.resolve()
    config.timestamp = args.run_id
    config.max_num_iterations = args.steps
    config.steps_per_save = args.checkpoint_every if args.checkpoint_every is not None else 250
    config.steps_per_eval_batch = config.steps_per_eval_image = config.steps_per_eval_all_images = 0
    config.save_only_latest_checkpoint = False
    config.vis = "tensorboard"
    config.viewer.quit_on_train_completion = True
    config.pipeline.datamanager.dataparser.max_eval_frames = None
    config.machine.seed = args.seed
    config.machine.num_devices = 1
    config.monitor_interval = args.monitor_every
    config.gradient_interval = args.gradient_every
    if args.pose_monitor_every is not None:
        config.pose_monitor_interval = args.pose_monitor_every
    config.pose_monitor_dir = args.pose_monitor_dir.resolve() if args.pose_monitor_dir else None
    config.quality_set = args.quality_set.resolve() if args.quality_set else None
    config.quality_dir = args.quality_dir.resolve() if args.quality_dir else None
    if args.quality_every is not None:
        config.quality_interval = args.quality_every
    model = {f.name: getattr(base.pipeline.model, f.name) for f in fields(SplatADModelConfig) if f.name != "_target"}
    manifest = args.manifest.resolve() if args.manifest else None
    if manifest:
        document = read_json(manifest)
        if document["scene_id"] != args.scene or not set(document["cameras"]) <= set(args.cameras):
            raise ValueError("Manifest scene and cameras must be included in this training run")
    model.update(implementation="torch", n_far_points=float(args.far_points),
                 max_num_seed_points=args.seed_points, num_downscales=args.downscales,
                 max_steps=args.schedule_steps)
    if args.variant == "SplatAD":
        config.pipeline.model = SplatADModelConfig(**model)
    else:
        config.pipeline.model = PedestrianModelConfig(
            **model, variant=args.variant, scene_id=args.scene,
            tracks_path=args.tracks.resolve() if args.tracks else None,
            body_path=args.body.resolve() if args.body else None,
            manifest_path=manifest, manifest_digest=sha256(manifest) if manifest else "",
            pedestrian_mcmc=bool(args.mcmc_cap), far_deformation=args.far_deformation,
            ground_path=args.ground.resolve() if args.ground else None,
            ground_digest=sha256(args.ground) if args.ground else "", ground_prefit=args.ground_prefit,
            joint_smooth_weight=args.joint_smooth_weight, contact_weight=args.contact_weight, slip_weight=args.slip_weight,
            person_rgb_weight=args.person_rgb_weight,
        )
        config.pipeline.model.actor_flip_probability = 0.0
        if args.mcmc_cap:
            config.pipeline.model.mcmc_cap_max = args.mcmc_cap
    config.pipeline.calc_fid_steps = ()
    datamanager = config.pipeline.datamanager
    datamanager.cache_images = datamanager.cache_lidars = "cpu"
    datamanager.dataparser.data = args.raw.resolve()
    datamanager.dataparser.sequence = args.scene
    datamanager.dataparser.cameras = tuple(args.cameras)
    if args.variant != "SplatAD":
        datamanager.dataparser.include_deformable_actors = True
    datamanager.dataparser.train_split_fraction = 1.0
    if args.variant in SMPL_VARIANTS:
        for name in GAUSSIAN_PARAMETERS:
            config.optimizers[f"smpl_{name}"] = copy.deepcopy(config.optimizers[name])
    if args.variant == "R3" and (args.mcmc_cap or args.far_deformation):
        for name in GAUSSIAN_PARAMETERS:
            config.optimizers[f"pedestrian_{name}"] = copy.deepcopy(config.optimizers[name])
        config.optimizers["pedestrian_deformation"] = {"optimizer": AdamOptimizerConfig(lr=1e-3), "scheduler": None}
    if args.variant in TRAINABLE_POSE_VARIANTS:
        for name, lr in zip(POSE_GROUPS, (3e-4, 2e-4, 2e-4), strict=True):
            config.optimizers[name] = {"optimizer": AdamOptimizerConfig(lr=lr), "scheduler": None}
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--scene", default="011")
    parser.add_argument("--variant", choices=("SplatAD", "R0", *SMPL_VARIANTS), default="R2")
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--body", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mcmc-cap", type=int, default=0, help="R3 MCMC cap for background and rigid non-pedestrian actors")
    parser.add_argument("--far-deformation", action="store_true", help="R3 local deformation for far pedestrian tracks")
    parser.add_argument("--ground", type=Path, help="R3 local ground planes from research.ground")
    parser.add_argument("--ground-prefit", action="store_true", help="Apply a bounded vertical fit before R3 optimization")
    parser.add_argument("--joint-smooth-weight", type=float, default=0.02)
    parser.add_argument("--contact-weight", type=float, default=0.1)
    parser.add_argument("--slip-weight", type=float, default=0.1)
    parser.add_argument("--person-rgb-weight", type=float, default=0.15, help="R3 RGB supervision in pedestrian cuboid crops")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", default="{timestamp}")
    parser.add_argument("--steps", type=int, default=200, help="Number of iterations to execute in this session")
    parser.add_argument("--schedule-steps", type=int, default=30001)
    parser.add_argument("--checkpoint-every", type=int, help="Checkpoint interval. New runs default to 250 steps.")
    parser.add_argument("--monitor-every", type=int, default=250)
    parser.add_argument("--gradient-every", type=int, default=25)
    parser.add_argument("--pose-monitor-every", type=int, help="RGB pose monitor interval, use 0 to disable")
    parser.add_argument("--pose-monitor-dir", type=Path, help="Dashboard artifact folder for live SMPL monitoring")
    parser.add_argument("--quality-set", type=Path, help="Frozen RGB examples shared with the SplatAD baseline")
    parser.add_argument("--quality-dir", type=Path, help="Dashboard artifact folder for live PSNR and SSIM")
    parser.add_argument("--quality-every", type=int, help="Optimizer steps between RGB measurements, use 0 to disable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-points", type=int, default=250000)
    parser.add_argument("--far-points", type=int, default=80000)
    parser.add_argument("--downscales", type=int, default=0,
                        help="Initial RGB downscaling stages. Use 0 for native training resolution.")
    parser.add_argument("--cameras", nargs="+", default=["front_right"])
    parser.add_argument("--resume", type=Path, help="Trusted config.yml from the existing run")
    parser.add_argument("--checkpoint", type=Path, help="Trusted checkpoint for resume")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required. In Colab select a GPU runtime and run bootstrap first.")
    if (args.steps < 1 or args.gradient_every < 1 or args.monitor_every < 0
            or (args.checkpoint_every is not None and args.checkpoint_every < 1)):
        parser.error("Invalid step intervals")
    if args.pose_monitor_every is not None and args.pose_monitor_every < 0:
        parser.error("The pose monitor interval must be nonnegative")
    if args.quality_every is not None and args.quality_every < 0:
        parser.error("The RGB quality interval must be nonnegative")
    if args.mcmc_cap < 0 or any(not 0 <= value < float("inf") for value in
                              (args.joint_smooth_weight, args.contact_weight, args.slip_weight, args.person_rgb_weight)):
        parser.error("MCMC cap and loss weights must be nonnegative")
    if (args.mcmc_cap or args.far_deformation or args.ground or args.ground_prefit) and (args.resume or args.variant != "R3"):
        parser.error("Set R3 research options when starting a new R3 run. Resume uses the saved configuration.")
    if args.ground_prefit and not args.ground:
        parser.error("--ground-prefit requires --ground")
    if args.resume:
        if not args.checkpoint:
            parser.error("--resume requires --checkpoint")
        config = yaml.unsafe_load(args.resume.read_text(encoding="utf-8"))
        if not isinstance(config, ResearchConfig):
            parser.error("Resume config was not produced by this preset")
        config.load_dir = None
        config.load_checkpoint = args.checkpoint.resolve()
        config.max_num_iterations = args.steps
        if args.raw is not None:
            config.pipeline.datamanager.dataparser.data = args.raw.resolve()
        if args.checkpoint_every is not None:
            config.steps_per_save = args.checkpoint_every
        if args.pose_monitor_every is not None:
            config.pose_monitor_interval = args.pose_monitor_every
        if args.pose_monitor_dir:
            config.pose_monitor_dir = args.pose_monitor_dir.resolve()
        if args.quality_set:
            config.quality_set = args.quality_set.resolve()
        if args.quality_dir:
            config.quality_dir = args.quality_dir.resolve()
        if args.quality_every is not None:
            config.quality_interval = args.quality_every
    else:
        if args.raw is None or (args.variant in SMPL_VARIANTS and not all((args.tracks, args.body, args.manifest))):
            parser.error("Supply --raw and, for R1/R2/R3, --tracks, --body and --manifest")
        config = make_config(args)
        existing = config.get_checkpoint_dir()
        if config.timestamp != "{timestamp}" and any(complete_checkpoint(path) for path in existing.glob("step-*.ckpt")):
            parser.error("Run ID already exists. Use --resume or choose another --run-id")
    os.environ.setdefault("WANDB_MODE", "disabled")
    launch(config)


if __name__ == "__main__":
    # YAML checkpoints need a module-qualified class, even with python -m.
    from research.train import main as entrypoint

    entrypoint()
