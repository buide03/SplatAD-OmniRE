"""Check raw SplatAD rendering, modality-specific pose updates and checkpoint replay."""
import argparse
from pathlib import Path

import torch

from .evaluate import camera_indices, camera_sample, lidar_sample, load_checkpoint
from .model import POSE_GROUPS
from .pose_monitor import POSE_ATTRS, drift, finite_number, gradient_stats
from .prepare import dump_json, read_json, sha256


def fixed_observation(manifest, model, sensor):
    rows = [r for r in manifest["observations"] if r["probe"]]
    if model.people:
        rows = [r for r in rows if r["uuid"] in model.person_index and r["smpl_valid"]]
    if sensor == "lidar":
        rows = [r for r in rows if r["raw_lidar_hits"] > 0]
    if not rows:
        raise ValueError(f"No eligible {sensor} probe in the frozen manifest")
    score = lambda r: r["bbox_height_px"] if sensor == "rgb" else r["raw_lidar_hits"]
    return max(rows, key=score)


def probe_sensor(config, pipeline, row, sensor):
    manager, model = pipeline.datamanager, pipeline.model
    if sensor == "rgb":
        idx = camera_indices(manager)[row["camera"], row["frame"]]
        instrument, image = camera_sample(manager, idx)
        batch = {"image": image}
        output_key = "rgb"
    else:
        instrument, batch = lidar_sample(manager, row["scan"], training=True)
        output_key = "depth"
    result = {"sensor": sensor, "uuid": row["uuid"], "camera": row["camera"],
              "frame": row["frame"], "scan": row["scan"], "raw_lidar_hits": row["raw_lidar_hits"]}
    model.eval()
    with torch.no_grad():
        output = model.get_outputs(instrument)[output_key]
        result.update({"shape": list(output.shape), "render_finite": bool(torch.isfinite(output).all())})
        reference = output.detach().clone()
    if config.pipeline.model.variant not in ("R2", "R3"):
        return result, instrument, output_key, reference

    model.train()
    groups = {name: value for name, value in model.get_param_groups().items() if name.startswith("smpl_")}
    optimizer = torch.optim.Adam([
        {"params": params, "lr": config.optimizers[name]["optimizer"].lr} for name, params in groups.items()
    ])
    params = [p for group in groups.values() for p in group]
    saved = [p.detach().clone() for p in params]
    person = model.people[model.person_index[row["uuid"]]]
    before = {attr: getattr(person, attr).detach().clone() for attr in POSE_ATTRS}
    try:
        outputs = model.get_outputs(instrument)
        losses = model.get_loss_dict(outputs, batch)
        loss = sum(losses.values())
        for parameter, grad in zip(params, torch.autograd.grad(loss, params, allow_unused=True), strict=True):
            parameter.grad = grad
        result["loss"] = finite_number(float(loss.detach()))
        result["gradient_groups"] = {name: gradient_stats(group) for name, group in groups.items()}
        lo, hi, _, visible = person._span(model.last_actor_time.to(person.times).reshape(()))
        indices = torch.stack((lo, hi)).unique()
        optimizer.step()
        with torch.no_grad():
            result["pose_visible_at_render_time"] = bool(visible)
            result["active_frames"] = person.frame_ids[indices].tolist()
            result["active_pose"] = {}
            for name, attr in zip(POSE_GROUPS, POSE_ATTRS, strict=True):
                parameter = getattr(person, attr)
                active = parameter[indices]
                result["active_pose"][name] = {
                    "gradient": gradient_stats([parameter], indices),
                    "delta": drift(active, before[attr][indices], attr != "translation"),
                }
        return result, instrument, output_key, reference
    finally:
        with torch.no_grad():
            for parameter, old in zip(params, saved, strict=True):
                parameter.copy_(old)
                parameter.grad = None
        model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, pipeline, checkpoint, step = load_checkpoint(args.config.resolve(), args.checkpoint)
    path = args.manifest or config.pipeline.model.manifest_path
    if not path:
        parser.error("Supply a manifest with fixed observations")
    manifest = read_json(path)
    if manifest["scene_id"] != config.pipeline.model.scene_id:
        parser.error("Manifest and checkpoint belong to different scenes")
    if config.pipeline.model.manifest_digest and config.pipeline.model.manifest_digest != sha256(path):
        parser.error("The manifest differs from the one used for training")
    pairs = [probe_sensor(config, pipeline, fixed_observation(manifest, pipeline.model, sensor), sensor)
             for sensor in ("rgb", "lidar")]
    trained = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pipeline.load_pipeline(trained["pipeline"], step, strict=True)
    with torch.no_grad():
        for result, instrument, output_key, reference in pairs:
            predicted = pipeline.model.get_outputs(instrument)[output_key]
            result["checkpoint_max_abs_difference"] = finite_number(float((predicted - reference).abs().max()))
            result["checkpoint_replay_matches"] = bool(torch.allclose(predicted, reference, atol=1e-5, rtol=1e-5))
    results = [pair[0] for pair in pairs]
    for result in results:
        pose = result.get("active_pose")
        result["passed"] = result["render_finite"] and result["checkpoint_replay_matches"]
        if pose is not None:
            result["passed"] = result["passed"] and result["pose_visible_at_render_time"] and all(
                group["gradient"]["state"] == "finite" and (group["delta"]["max"] or 0) > 0
                for group in pose.values()
            )
    dump_json(args.output, {"variant": config.pipeline.model.variant, "step": step, "tests": results})
    print(args.output)
    if not all(result["passed"] for result in results):
        raise SystemExit("At least one RGB/LiDAR check failed. Inspect the probe JSON.")


if __name__ == "__main__":
    main()
