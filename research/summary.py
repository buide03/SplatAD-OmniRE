"""Compare SplatAD, R2, R3 and OmniRe on the frozen RGB samples at the steps every run measured."""
import argparse
import csv
from pathlib import Path

from .prepare import dump_json, read_json
from .quality import SCOPES

ORDER = ("SplatAD", "R2", "R3", "OmniRe")


def summarize(reports):
    runs = {}
    for report in reports:
        if report["variant"] in runs:
            raise ValueError(f"Two reports for {report['variant']}")
        runs[report["variant"]] = report
    if len({(r["scene"], r["quality_sha256"]) for r in runs.values()}) != 1:
        raise ValueError("Reports must share the scene and the frozen RGB samples")
    variants = sorted(runs, key=lambda name: (ORDER.index(name) if name in ORDER else len(ORDER), name))
    by_step = {name: {row["step"]: row["scopes"] for row in runs[name]["steps"]} for name in variants}
    shared = sorted(set.intersection(*(set(steps) for steps in by_step.values())))
    rows = []
    for step in shared:
        for scope in SCOPES:
            baseline = by_step.get("SplatAD", {}).get(step, {}).get(scope, {})
            for name in variants:
                values = by_step[name][step][scope]
                rows.append({
                    "step": step, "scope": scope, "variant": name, "samples": values["samples"],
                    "psnr_db": values["psnr_db"], "ssim": values["ssim"],
                    "delta_psnr_db": None if baseline.get("psnr_db") is None or values["psnr_db"] is None
                    else values["psnr_db"] - baseline["psnr_db"],
                    "delta_ssim": None if baseline.get("ssim") is None or values["ssim"] is None
                    else values["ssim"] - baseline["ssim"],
                })
    return {"scene": next(iter(runs.values()))["scene"], "variants": variants, "steps": shared,
            "run_ids": {name: runs[name]["run_id"] for name in variants}, "rows": rows}


def markdown(summary, step=None):
    step = summary["steps"][-1] if step is None else step
    lines = [f"Cảnh {summary['scene']} · bước {step} · Δ so với SplatAD",
             "| Vùng | " + " | ".join(summary["variants"]) + " |",
             "| --- |" + " ---: |" * len(summary["variants"])]
    for scope in SCOPES:
        cells = []
        for name in summary["variants"]:
            row = next(r for r in summary["rows"] if r["step"] == step and r["scope"] == scope and r["variant"] == name)
            if row["psnr_db"] is None:
                cells.append("—")
                continue
            delta = "" if name == "SplatAD" or row["delta_psnr_db"] is None else f" ({row['delta_psnr_db']:+.2f})"
            cells.append(f"{row['psnr_db']:.2f} dB{delta} / {row['ssim']:.3f}")
        lines.append(f"| {scope} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True, help="quality.json of each run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize([read_json(path) for path in args.reports])
    if not summary["steps"]:
        parser.error("The runs share no measured step")
    args.output.mkdir(parents=True, exist_ok=True)
    dump_json(args.output / "summary.json", summary)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    table = markdown(summary)
    (args.output / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
