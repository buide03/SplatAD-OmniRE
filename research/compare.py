"""Compute paired R0/R1/R2 summaries on identical SMPL-covered observations."""
import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

from .prepare import dump_json, read_json

METRICS = ("mae", "psnr_db", "ssim", "lpips", "range_mae_m", "reprojection_mae_px")


def key(row):
    return row["scene_id"], row["uuid"], row["camera"], row["frame"]


def paired(rows_a, rows_b, metric, bin_name, common_keys=None):
    other = {key(row): row for row in rows_b}
    people = defaultdict(dict)
    crop_count = 0
    for a in rows_a:
        if common_keys is not None and key(a) not in common_keys:
            continue
        b = other.get(key(a))
        if not b or not (a["smpl_valid"] and b["smpl_valid"] and a["bin"] == b["bin"] == bin_name):
            continue
        if not a.get("comparison_eligible", True) or not b.get("comparison_eligible", True):
            continue
        if metric == "range_mae_m" and a["lidar_hits"] != b["lidar_hits"]:
            raise ValueError(f"Unmatched LiDAR ray counts for {key(a)}")
        if a.get(metric) is None or b.get(metric) is None:
            continue
        observation = a["scan"] if metric == "range_mae_m" else (a["camera"], a["frame"])
        people[a["scene_id"], a["uuid"]][observation] = (a[metric], b[metric], a["lidar_hits"])
        crop_count += 1
    per_scene = defaultdict(list)
    for (scene, _), samples in people.items():
        per_scene[scene].append(tuple(mean(v[i] for v in samples.values()) for i in (0, 1)))
    scene_scores = {scene: tuple(mean(s[i] for s in scores) for i in (0, 1))
                    for scene, scores in per_scene.items()}
    values = [mean(score[i] for score in scene_scores.values()) for i in (0, 1)] if scene_scores else [None, None]
    return {
        "bin": bin_name, "metric": metric, "a": values[0], "b": values[1],
        "delta_b_minus_a": values[1] - values[0] if scene_scores else None,
        "n_scene": len(scene_scores), "n_uuid": len(people), "n_crop": crop_count,
        "n_samples": sum(len(samples) for samples in people.values()),
        "lidar_hits": sum(row[2] for samples in people.values() for row in samples.values()) if metric == "range_mae_m" else None,
        "scene_scores": {scene: {"a": a, "b": b, "delta_b_minus_a": b - a}
                         for scene, (a, b) in scene_scores.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = defaultdict(list)
    manifests = {}
    coverage = {}
    for folder in args.runs:
        run = read_json(folder / "run.json")
        metrics = read_json(folder / run["metrics"])
        scene = run["scene"]
        hashes = manifests.setdefault(scene, set())
        hashes.add(run["manifest_sha256"])
        if len(hashes) > 1:
            parser.error(f"Runs for {scene} use different manifests")
        variant = run["variant"] if run["mode"] == "checkpoint" else run["variant"] + "/velocity-off"
        if any(r["scene_id"] == scene for r in variants[variant]):
            parser.error(f"Multiple runs for {scene}/{variant}")
        variants[variant].extend(metrics["observations"])
        coverage[scene] = metrics["coverage"]
    comparisons = []
    for a, b in combinations(sorted(variants), 2):
        for bin_name in ("near", "mid", "far"):
            for metric in METRICS:
                common_keys = None
                baseline_comparison = all(name in variants for name in ("R0", "R1", "R2")) and "/" not in a + b
                if baseline_comparison and metric != "reprojection_mae_px":
                    common_keys = set.intersection(*[
                        {key(row) for row in variants[name] if row["smpl_valid"]
                         and row.get("comparison_eligible", True) and row.get(metric) is not None}
                        for name in ("R0", "R1", "R2")
                    ])
                comparisons.append({
                    "comparison": f"{a} → {b}", **paired(variants[a], variants[b], metric, bin_name, common_keys)
                })
    args.output.mkdir(parents=True, exist_ok=True)
    dump_json(args.output / "comparison.json", {"coverage": coverage, "comparisons": comparisons})
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["comparison", "bin", "metric", "a", "b", "delta_b_minus_a", "n_scene",
                   "n_uuid", "n_crop", "n_samples", "lidar_hits"]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparisons)


if __name__ == "__main__":
    main()
