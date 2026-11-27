"""Track RGB fit on fixed observations and isolate the effect of one SMPL pose update."""
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .geometry import matrix_quat
from .evaluate import camera_indices, camera_sample, crop_bounds, overlay, project, reprojection, to_image
from .prepare import dump_json, front_pose_summary, read_json

POSE_ATTRS = ("translation", "root", "body")


def finite_number(number):
    return number if math.isfinite(number) else None


def gradient_stats(parameters, indices=None):
    values = [p.grad.detach() for p in parameters if p.grad is not None]
    if indices is not None:
        values = [value[indices] for value in values]
    return describe_gradients(values)


def describe_gradients(values):
    if not values:
        return {"state": "none", "norm": None, "parameters_with_grad": 0}
    norm = torch.linalg.vector_norm(torch.stack([g.float().norm() for g in values])).item()
    state = "non_finite" if not math.isfinite(norm) else ("zero" if norm == 0 else "finite")
    return {"state": state, "norm": finite_number(norm), "parameters_with_grad": len(values)}


def drift(current, reference, rotation):
    if rotation:
        cos = (F.normalize(current, dim=-1) * F.normalize(reference, dim=-1)).sum(-1, keepdim=True)
        aligned = torch.where(cos < 0, -reference, reference)
        chord = (F.normalize(current, dim=-1) - F.normalize(aligned, dim=-1)).norm(dim=-1)
        displacement = 4 * torch.rad2deg((chord / 2).clamp(0, 1).asin())
    else:
        displacement = (current - reference).norm(dim=-1) * 100
    return {"mean": finite_number(displacement.mean().item()), "max": finite_number(displacement.max().item()),
            "unit": "deg" if rotation else "cm"}


def select_observations(manifest, model, cameras):
    rows = [
        row for row in manifest["observations"]
        if (row["usable"] or (row.get("pose_monitor") and row.get("limbs_visible") is False))
        and row["smpl_valid"] and row["uuid"] in model.person_index
        and (row["camera"], row["frame"]) in cameras
    ]
    explicit = [row for row in rows if row.get("pose_monitor")]
    preferred = explicit or [row for row in rows if row.get("silhouette_xy") or row.get("keypoints")]
    candidates = preferred or [row for row in rows if row["probe"]] or rows
    return sorted(candidates, key=lambda row: (
        -bool(row.get("silhouette_xy")), -bool(row.get("keypoints")), row.get("limbs_visible") is not True,
        -row["bbox_height_px"], row["camera"], row["frame"], row["uuid"],
    ))[:2]


def track_observations(manifest, people, cameras, taken=()):
    """One clearly visible observation per SMPL pedestrian, so every mesh is followed during training."""
    best = {}
    for row in manifest["observations"]:
        if (not row["usable"] or not row["smpl_valid"] or row["uuid"] not in people or row["uuid"] in taken
                or (row["camera"], row["frame"]) not in cameras):
            continue
        rank = (row.get("limbs_visible") is not False, row["bbox_height_px"], -row["frame"], row["camera"])
        if row["uuid"] not in best or rank > best[row["uuid"]][0]:
            best[row["uuid"]] = (rank, row)
    return [best[uuid][1] for uuid in sorted(best)]


def pose_drift(current, initial):
    """Displacement of every stored pose from its initialization: root translation, root and joint rotations."""
    return {name: drift(current[name].detach(), initial[name].detach(), name != "translation") for name in current}


def polygon_mask(shape, polygon):
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def mesh_silhouette(shape, projection, faces):
    import cv2

    xy, forward = projection
    triangles = faces[forward[faces].all(-1) & torch.isfinite(xy[faces]).all(dim=(1, 2))]
    coordinates = np.rint(xy[triangles].clamp(-32000, 32000).cpu().numpy()).astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    for triangle in coordinates:
        cv2.fillConvexPoly(mask, triangle, 1)
    return mask.astype(bool)


def iou(prediction, reference):
    union = np.count_nonzero(prediction | reference)
    return float(np.count_nonzero(prediction & reference) / union) if union else None


@torch.no_grad()
def compare_meshes(gt, prediction, row, initial, current, faces, include_initial=True):
    """Overlay the initial and current SMPL mesh on the original RGB and score the person region.

    ``gt`` and ``prediction`` are HxWx3 images in [0, 1]. ``initial`` and ``current`` are
    ``(projection, visible)`` pairs, where a projection is ``(xy, in_front)`` from ``evaluate.project``.
    """
    import cv2

    if gt.shape != prediction.shape:
        raise ValueError(f"Monitor RGB shapes differ: {gt.shape} and {prediction.shape}")
    height, width = gt.shape[:2]
    x1, y1, x2, y2 = crop_bounds(row["bbox_xyxy"], height, width)
    roi = np.s_[y1:y2, x1:x2]
    gt_rgb = to_image(gt)
    diff = (prediction - gt).abs()
    (xy_initial, initial_visible), (xy_current, visible) = initial, current
    ref = row.get("silhouette_xy")
    metrics = {"mesh_visible": visible, "silhouette_iou": None, "silhouette_initial_iou": None}
    reference_image = gt_rgb.copy()
    if ref:
        reference = polygon_mask((height, width), ref)
        if reference.any():
            metrics.update(roi_source="manual_silhouette", roi_pixels=int(reference.sum()),
                           rgb_mae=finite_number(float(diff[torch.as_tensor(reference, device=diff.device)].mean())))
            if visible:
                metrics["silhouette_iou"] = iou(mesh_silhouette(reference.shape, xy_current, faces), reference)
            if include_initial and initial_visible:
                metrics["silhouette_initial_iou"] = iou(mesh_silhouette(reference.shape, xy_initial, faces), reference)
            cv2.polylines(reference_image, [np.rint(ref).astype(np.int32)], True, (70, 213, 255), 2, cv2.LINE_AA)
    if "rgb_mae" not in metrics:
        bx1, by1, bx2, by2 = crop_bounds(row["bbox_xyxy"], height, width, minimum=1)
        metrics.update(roi_source="fixed_bbox_crop", roi_pixels=(bx2 - bx1) * (by2 - by1),
                       rgb_mae=finite_number(float(diff[by1:by2, bx1:bx2].mean())))
    labels = row.get("keypoints", {})
    for point in labels.values():
        cv2.circle(reference_image, tuple(int(round(value)) for value in point["xy"]), 3,
                   (70, 213, 255), 1, cv2.LINE_AA)
    point_errors = reprojection(labels, (None, None, xy_initial, xy_current, initial_visible and visible))
    if point_errors["keypoint_count"] != len(labels):
        point_errors.update(reprojection_mae_px=None, reprojection_initial_mae_px=None)
    metrics.update(point_errors)
    panels = {
        "reference": reference_image[roi].copy(),
        "current": overlay(reference_image, xy_current, visible, faces, (100, 246, 181))[roi].copy(),
        "render": to_image(prediction)[roi].copy(),
        "error": to_image(diff * 3)[roi].copy(),
    }
    if include_initial:
        panels["initial"] = overlay(reference_image, xy_initial, initial_visible, faces, (252, 192, 89))[roi].copy()
    return metrics, panels


@torch.no_grad()
def measure_rgb(model, camera, image, person, row, include_initial=True):
    """Use the same native-resolution reference pixels and projection for each snapshot."""
    outputs = model.get_camera_outputs(camera)
    prediction = outputs["rgb"]
    gt = model.composite_with_background(model.get_gt_img(image), outputs["background"])
    params = (outputs["world_to_camera"], camera.get_intrinsics_matrices()[0])
    capture_time = outputs["capture_time"]
    initial, initial_visible = person.mesh_at(capture_time, initial=True)
    current, current_visible = person.mesh_at(capture_time)
    visible = bool(current_visible)
    metrics, panels = compare_meshes(gt, prediction, row, (project(initial, *params), bool(initial_visible)),
                                     (project(current, *params), visible), person.faces, include_initial)
    if visible and hasattr(person, "constraints"):
        metrics.update(person.constraints.measure(person, capture_time))
    return metrics, panels


def pose_only_rgb_step(model, person, fetch_batch, learning_rates, measure_after):
    """Use the scene RGB loss for a temporary Adam step on active rows of one person's pose."""
    parameters = [getattr(person, name) for name in POSE_ATTRS]
    if not all(p.requires_grad for p in parameters):
        return {"status": "frozen"}, {}
    was_training = model.training
    state_keys = ("last_actor_time", "last_sensor", "last_rgb_index", "last_size", "info")
    saved_state = {name: getattr(model, name) for name in state_keys if hasattr(model, name)}
    devices = [parameters[0].device] if parameters[0].is_cuda else []
    originals = None
    indices = None

    def total_rgb_loss():
        camera, batch = fetch_batch()
        outputs = model.get_camera_outputs(camera)
        return sum(model.get_loss_dict(outputs, batch).values())

    try:
        # Matching randomness makes both loss readings comparable and preserves training RNG.
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            cpu_rng = torch.get_rng_state()
            cuda_rng = [torch.cuda.get_rng_state(device) for device in devices]
            model.train()
            loss = total_rgb_loss()
            result = {"loss_before": finite_number(float(loss.detach()))}
            lo, hi, _, visible = person._span(model.last_actor_time.to(person.times).reshape(()))
            if not bool(visible):
                return {**result, "status": "pose_unavailable"}, {}
            indices = torch.stack((lo, hi)).unique()
            if result["loss_before"] is None:
                return {**result, "status": "non_finite_loss"}, {}
            grads = torch.autograd.grad(loss, parameters, allow_unused=True) if loss.requires_grad else (None,) * 3
            active = [None if grad is None else grad[indices].detach() for grad in grads]
            result["groups"] = {name: describe_gradients([grad] if grad is not None else [])
                                for name, grad in zip(POSE_ATTRS, active, strict=True)}
            states = {group["state"] for group in result["groups"].values()}
            if "non_finite" in states or "finite" not in states:
                status = "non_finite_gradient" if "non_finite" in states else "no_pose_gradient"
                return {**result, "status": status}, {}

            originals = [p[indices].detach().clone() for p in parameters]
            proposals = [nn.Parameter(old.clone()) for old in originals]
            optimizer = torch.optim.Adam([{"params": [p], "lr": rate}
                                          for p, rate in zip(proposals, learning_rates, strict=True)])
            for proposal, grad in zip(proposals, active, strict=True):
                proposal.grad = grad
            optimizer.step()
            with torch.no_grad():
                for i, (name, parameter, proposal) in enumerate(zip(POSE_ATTRS, parameters, proposals, strict=True)):
                    result["groups"][name]["delta"] = drift(proposal, originals[i], name != "translation")
                    parameter.index_copy_(0, indices, proposal)
                torch.set_rng_state(cpu_rng)
                for device, rng in zip(devices, cuda_rng, strict=True):
                    torch.cuda.set_rng_state(rng, device)
                result["loss_after"] = finite_number(float(total_rgb_loss()))
                model.eval()
                result["after"], panels = measure_after()
            return {**result, "status": "measured", "active_frames": person.frame_ids[indices].tolist()}, panels
    finally:
        if originals is not None:
            with torch.no_grad():
                for parameter, original in zip(parameters, originals, strict=True):
                    parameter.index_copy_(0, indices, original)
        model.train(was_training)
        for name in state_keys:
            if name in saved_state:
                setattr(model, name, saved_state[name])
            elif hasattr(model, name):
                delattr(model, name)


class PoseMonitor:
    def __init__(self, pipeline, manifest, output, run_id, start_step, manifest_digest):
        self.model = pipeline.model
        self.manager = pipeline.datamanager
        self.output = Path(output)
        self.record = {"scene": manifest["scene_id"], "variant": self.model.config.variant,
                       "mode": "pose_monitor", "run_id": run_id, "videos": [], "monitor": "monitor.json",
                       "pose_summary": front_pose_summary(manifest)}
        cameras = camera_indices(self.manager)
        rows = select_observations(manifest, self.model, cameras)
        tracked = track_observations(manifest, self.model.person_index, cameras, {row["uuid"] for row in rows})
        self.observations = {
            f"{row['camera']}/{row['frame']}/{row['uuid']}": {
                **row, "image_index": cameras[row["camera"], row["frame"]], "probe_update": i < len(rows)}
            for i, row in enumerate([*rows, *tracked])
        }
        self.watched_uuids = {row["uuid"] for row in rows}
        self.watched_indices = {row["image_index"] for row in self.observations.values()}
        self.manifest_digest = manifest_digest
        self.steps = []
        self.actual = {}
        previous = self.output / self.record["monitor"]
        if start_step and previous.exists():
            stored = read_json(previous)
            if stored["manifest_sha256"] != manifest_digest or stored["observations"] != self.observations:
                raise ValueError("Pose monitor output belongs to different frozen observations")
            self.steps = [step for step in stored["steps"] if step["step"] <= start_step]
            for step in self.steps:
                for key, row in step["results"].items():
                    if row.get("actual_rgb_step"):
                        self.actual[key] = row["actual_rgb_step"]
        if self.observations:
            self.publish()

    def publish(self):
        dump_json(self.output / self.record["monitor"], {
            "manifest_sha256": self.manifest_digest, "observations": self.observations, "steps": self.steps,
        })
        dump_json(self.output / "run.json", {**self.record, "step": self.steps[-1]["step"] if self.steps else 0,
                                           "snapshots": len(self.steps)})

    def observe_training_step(self, row):
        if row["sensor"] != "rgb" or row.get("image_index") not in self.watched_indices:
            return
        for key, desired in self.observations.items():
            pose = row["pose"].get(desired["uuid"])
            if desired["image_index"] == row["image_index"] and pose is not None:
                self.actual[key] = {"step": row["step"] + 1, "scene_loss": row["scene_loss"], "groups": pose}

    def capture(self, step, learning_rates):
        was_training = self.model.training
        results = {}
        try:
            for i, (key, row) in enumerate(self.observations.items()):
                person = self.model.people[self.model.person_index[row["uuid"]]]
                index = row["image_index"]

                def fetch_batch():
                    camera, image = camera_sample(self.manager, index)
                    batch = {**self.manager.cached_train[index], "image": image}
                    if "mask" in batch:
                        height, width = image.shape[:2]
                        batch["mask"] = batch["mask"][:height, :width].to(image.device)
                    return camera, batch

                def measure(include_initial=True):
                    camera, batch = fetch_batch()
                    config = self.model.config
                    velocity_config = self.model.camera_velocity_optimizer.config
                    use_camopt, velocity_enabled = config.use_camopt_in_eval, velocity_config.enabled
                    config.use_camopt_in_eval = False
                    velocity_config.enabled = False
                    try:
                        return measure_rgb(self.model, camera, batch["image"], person, row, include_initial)
                    finally:
                        config.use_camopt_in_eval = use_camopt
                        velocity_config.enabled = velocity_enabled

                self.model.eval()
                values, images = measure()
                if learning_rates and row.get("probe_update", True):
                    update, updated_images = pose_only_rgb_step(self.model, person, fetch_batch, learning_rates,
                                                                lambda: measure(include_initial=False))
                else:
                    update, updated_images = {"status": "tracked" if learning_rates else "frozen"}, {}
                if "current" in updated_images:
                    images["after_probe"] = updated_images["current"]
                    images["render_after_probe"] = updated_images["render"]
                written = {}
                for name, image in images.items():
                    relative = Path(f"step-{step:06d}") / f"observation-{i + 1}" / f"{name}.png"
                    target = self.output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(target)
                    written[name] = relative.as_posix()
                results[key] = {"metrics": values, "panels": written, "pose_only_rgb_probe": update,
                                "actual_rgb_step": self.actual.get(key)}
                probe_iou = update.get("after", {}).get("silhouette_iou")
                print(f"Pose monitor {step} {row['camera']}/{row['frame']} {row['uuid']}: "
                      f"ROI MAE={values['rgb_mae']} mesh IoU={values['silhouette_iou']} "
                      f"RGB pose probe={update['status']} IoU after={probe_iou}", flush=True)
        finally:
            self.model.train(was_training)
        tracks = {}
        with torch.no_grad():
            for node in self.model.people:
                current = {name: getattr(node, name) for name in POSE_ATTRS}
                tracks[node.uuid] = pose_drift(current, {name: getattr(node, f"initial_{name}") for name in POSE_ATTRS})
                if getattr(node, "box_at", None) is not None:
                    # How far the learned cuboid carrying this person moved from its annotation.
                    rotation, position = node.box_at(node.times)
                    tracks[node.uuid]["box_translation"] = drift(position, node.anchor_position, False)
                    tracks[node.uuid]["box_rotation"] = drift(matrix_quat(rotation), matrix_quat(node.anchor_rotation), True)
        self.steps = [record for record in self.steps if record["step"] < step]
        self.steps.append({"step": step, "results": results, "tracks": tracks})
        self.publish()
