"""Diagnose R3 deformable pedestrians that render as a sliding rigid block.

For each deformable pedestrian of a checkpoint this measures:
  * whether the OmniRe deformation MLP is active at this step;
  * how much the MLP offsets move the Gaussians over time (a walking person needs leg offsets that
    change by tens of centimetres; a constant offset is a rigid block);
  * the Gaussians the node renders with (count, opacity, size, spread inside the box);
  * how large the person is in the training images;
  * on the largest training view: the gradient the image loss sends to the node, compared with the
    gradient the same pixels would send if they were weighted like a full image, and how much the
    MLP changes the rendered crop compared with switching it off.
"""
import argparse
from pathlib import Path

import torch

from .evaluate import camera_indices, camera_sample, load_checkpoint
from .prepare import dump_json, read_json

VISIBLE_OPACITY = 0.05
LEG_FRACTION = 0.35  # Lowest part of the box height treated as legs.
STATIC_LEG_CM = 2.0  # Leg offsets that vary less than this over time do not walk.


def quantiles(values, scale=1.0):
    values = values.detach().float().flatten().cpu()
    if not len(values):
        return None
    return {f"p{q}": round(float(values.quantile(q / 100)) * scale, 4) for q in (10, 50, 90)}


@torch.no_grad()
def gaussian_stats(node):
    opacity = node.opacities.sigmoid().flatten()
    size = node.size.detach()
    largest = node.scales.exp().max(dim=-1).values
    inside = (node.means.abs() <= size / 2).all(dim=-1)
    return {
        "gaussians": len(opacity),
        "visible": int((opacity > VISIBLE_OPACITY).sum()),
        "opaque": int((opacity > 0.5).sum()),
        "opacity": quantiles(opacity),
        "largest_axis_cm": quantiles(largest, 100),
        "inside_box": round(float(inside.float().mean()), 4),
        "box_size_m": [round(float(v), 3) for v in size],
    }


@torch.no_grad()
def deformation_stats(model, node):
    """Offsets of the MLP at every trajectory time the person is present, in centimetres."""
    field = model.deformation_field
    times = node.scene_times
    _, present = model.dynamic_actors.get_boxes2world(times, flatten=False)
    present = present.reshape(len(times), -1)[:, node.actor_index].bool()
    times = times[present.to(times.device)]
    canonical = node.means.detach()
    visible = node.opacities.sigmoid().flatten() > VISIBLE_OPACITY
    if field is None or len(times) < 2 or not visible.any():
        return {"frames": int(len(times)), "visible": int(visible.sum()), "measured": False}
    centers = 2 * canonical / node.size[2].clamp_min(1e-6)
    offsets, turns = [], []
    for time in times:
        delta, rotation = field(centers, node.normalized_time(time), node.embedding)
        offsets.append(delta[visible])
        turns.append(rotation[visible].norm(dim=-1))
    offsets = torch.stack(offsets)  # [time, gaussian, 3]
    spread = offsets.std(dim=0).norm(dim=-1)
    legs = canonical[visible, 2] < node.size[2] * (LEG_FRACTION - 0.5)
    result = {
        "frames": int(len(times)),
        "visible": int(visible.sum()),
        "measured": True,
        "offset_cm": quantiles(offsets.norm(dim=-1), 100),
        "temporal_std_cm": quantiles(spread, 100),
        "temporal_range_cm": quantiles((offsets.amax(dim=0) - offsets.amin(dim=0)).norm(dim=-1), 100),
        "frame_to_frame_cm": quantiles((offsets[1:] - offsets[:-1]).norm(dim=-1), 100),
        "rotation_delta": quantiles(torch.stack(turns)),
        "leg_gaussians": int(legs.sum()),
        "leg_temporal_std_cm": quantiles(spread[legs], 100) if legs.any() else None,
    }
    leg_std = result["leg_temporal_std_cm"] or result["temporal_std_cm"]
    result["walks"] = bool(leg_std["p90"] >= STATIC_LEG_CM)
    return result


def observation_stats(rows):
    usable = [r for r in rows if r["usable"]]
    heights = torch.tensor([float(r["bbox_height_px"]) for r in usable]) if usable else torch.zeros(0)
    bins = {}
    for row in usable:
        bins[row["bin"]] = bins.get(row["bin"], 0) + 1
    return {
        "observations": len(rows),
        "usable": len(usable),
        "frames": len({r["frame"] for r in usable}),
        "bins": bins,
        "bbox_height_px": quantiles(heights),
        "distance_m": quantiles(torch.tensor([float(r["distance_m"]) for r in usable])) if usable else None,
    }


def node_parameters(node):
    return {"gaussians": [node.features_dc, node.features_rest, node.opacities, node.scales, node.quats],
            "embedding": [node.embedding]}


def gradient_norms(loss, groups):
    names = list(groups)
    flat = [p for name in names for p in groups[name]]
    grads = torch.autograd.grad(loss, flat, retain_graph=True, allow_unused=True)
    norms, start = {}, 0
    for name in names:
        part = [g for g in grads[start:start + len(groups[name])] if g is not None]
        start += len(groups[name])
        norms[name] = float(torch.cat([g.flatten() for g in part]).norm()) if part else 0.0
    return norms


def psnr(a, b):
    return float(-10 * (a - b).square().mean().clamp_min(1e-12).log10())


def view_test(pipeline, row, node=None, person=None):
    """Full-image L1 against the same L1 averaged over the person's box only."""
    from .pose_monitor import POSE_ATTRS

    manager, model = pipeline.datamanager, pipeline.model
    camera, image = camera_sample(manager, camera_indices(manager)[row["camera"], row["frame"]])
    gt = model.get_gt_img(image)
    rendered = model.get_outputs(camera)["rgb"]
    if rendered.shape != gt.shape:
        raise ValueError(f"Render {tuple(rendered.shape)} differs from image {tuple(gt.shape)}")
    height, width = gt.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in row["bbox_xyxy"])
    x1, x2 = max(0, x1), min(width, max(x2, x1 + 1))
    y1, y2 = max(0, y1), min(height, max(y2, y1 + 1))
    roi = (slice(y1, y2), slice(x1, x2))
    if node is not None:
        groups = {**node_parameters(node), "network": list(model.deformation_field.parameters())}
    else:
        groups = {attr: [getattr(person, attr)] for attr in POSE_ATTRS}
    full = gradient_norms((rendered - gt).abs().mean(), groups)
    crop = gradient_norms((rendered[roi] - gt[roi]).abs().mean(), groups)
    result = {
        "camera": row["camera"], "frame": row["frame"], "bbox_height_px": row["bbox_height_px"],
        "crop_fraction": round((y2 - y1) * (x2 - x1) / (height * width), 6),
        "grad_full_image": full, "grad_crop_only": crop,
        "crop_psnr_db": round(psnr(rendered[roi].detach(), gt[roi]), 3),
    }
    if node is not None:
        start = model.config.deform_start
        try:
            model.config.deform_start = 10 ** 9
            with torch.no_grad():
                rigid = model.get_outputs(camera)["rgb"]
        finally:
            model.config.deform_start = start
        result["crop_psnr_without_mlp_db"] = round(psnr(rigid[roi], gt[roi]), 3)
        result["mlp_pixel_change"] = round(float((rigid[roi] - rendered[roi].detach()).abs().mean()), 5)
    return result


def largest_row(rows, cameras):
    usable = [r for r in rows if r["usable"] and (r["camera"], r["frame"]) in cameras]
    return max(usable, key=lambda r: r["bbox_height_px"]) if usable else None


def verdict(person):
    reasons = []
    motion = person["deformation"]
    if not motion.get("measured"):
        reasons.append("mlp_not_measured")
    elif not motion["walks"]:
        reasons.append("offsets_static_over_time")
    view = person.get("view")
    if view:
        if view["crop_fraction"] < 0.002:
            reasons.append("tiny_in_image")
        if view.get("mlp_pixel_change", 1) < 0.002:
            reasons.append("mlp_barely_changes_pixels")
    if person["gaussians"]["visible"] < 200:
        reasons.append("few_visible_gaussians")
    return reasons


def diagnose(pipeline, manifest):
    model = pipeline.model
    pipeline.eval()
    cameras = camera_indices(pipeline.datamanager)
    rows = {}
    for row in manifest["observations"]:
        rows.setdefault(row["uuid"], []).append(row)
    report = {
        "step": int(getattr(model, "step", -1)),
        "deform_start": int(model.config.deform_start),
        "mlp_active": model.deformation_field is not None and getattr(model, "step", -1) >= model.config.deform_start,
        "routes": [route for route in model.actor_routes if route.get("label") in ("Pedestrian", "Pedestrian with Object")
                   or route["representation"] != "rigid"],
        "deformable": [],
    }
    for node in model.deformable_people:
        route = model.actor_routes[node.actor_index]
        person = {"uuid": route["uuid"], "route": route, "gaussians": gaussian_stats(node),
                  "deformation": deformation_stats(model, node),
                  "observations": observation_stats(rows.get(route["uuid"], []))}
        row = largest_row(rows.get(route["uuid"], []), cameras)
        if row is not None:
            person["view"] = view_test(pipeline, row, node=node)
        person["verdict"] = verdict(person)
        report["deformable"].append(person)
    smpl = [(largest_row(rows.get(uuid, []), cameras), model.people[index]) for uuid, index in model.person_index.items()]
    smpl = [(row, person) for row, person in smpl if row is not None]
    if smpl:
        row, person = max(smpl, key=lambda pair: pair[0]["bbox_height_px"])
        report["smpl_reference"] = {"uuid": row["uuid"], "view": view_test(pipeline, row, person=person)}
    return report


def summary(report):
    lines = [f"step {report['step']}, MLP {'active' if report['mlp_active'] else 'OFF'} "
             f"(deform_start {report['deform_start']}), deformable pedestrians: {len(report['deformable'])}"]
    for person in report["deformable"]:
        motion, gauss, seen = person["deformation"], person["gaussians"], person["observations"]
        legs = motion.get("leg_temporal_std_cm") or motion.get("temporal_std_cm") or {}
        view = person.get("view") or {}
        lines.append(
            f"- {person['uuid'][:8]} seed={person['route'].get('seed')} gaussians={gauss['gaussians']} "
            f"visible={gauss['visible']} box_px_p50={(seen['bbox_height_px'] or {}).get('p50')} "
            f"leg_std_cm_p90={legs.get('p90')} crop={view.get('crop_fraction')} "
            f"psnr={view.get('crop_psnr_db')} no_mlp={view.get('crop_psnr_without_mlp_db')} "
            f"-> {', '.join(person['verdict']) or 'ok'}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, pipeline, _, _ = load_checkpoint(args.config.resolve(), args.checkpoint)
    if config.pipeline.model.variant != "R3":
        parser.error("Only R3 checkpoints have deformable pedestrians")
    report = diagnose(pipeline, read_json(args.manifest))
    dump_json(args.output, report)
    print(summary(report))
    print(args.output)


if __name__ == "__main__":
    main()
