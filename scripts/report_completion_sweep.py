"""Format a completion-sweep threshold_metrics.json as per-fault tables.

Reads the output of scripts.sweep_stage1_stage2_completion and prints (and
saves as report.md) one table per stage with the faulty baseline next to the
model at the stage's selected threshold, split into overall, total loss, FOV
loss and fog. `--full` adds every threshold for every fault.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGES = (
    ("stage1", "Stage I - radar support (all clean LiDAR within 3.2 m of radar)"),
    ("stage1_active", "Stage I - active cells only (generated, radar-supported cells)"),
    ("stage2", "Stage II - full ROI after completion"),
)
SCOPES = (("overall", "Overall"), ("total_loss", "Total loss"),
          ("fov_filter", "FOV loss"), ("fog_sim", "Fog"))
METRICS = (("iou_exact", "IoU"), ("iou_at_0_2m", "IoU@0.2m"),
           ("f1_exact", "F1"), ("f1_at_0_2m", "F1@0.2m"))


def _num(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def _pair(faulty, model, digits=3):
    return f"{_num(faulty, digits)} -> {_num(model, digits)}"


def _row_at(scope: dict, threshold: float) -> dict:
    for row in scope["rows"]:
        if abs(row["threshold"] - threshold) < 1e-9:
            return row
    raise KeyError(f"Threshold {threshold} not in sweep")


def _chamfer(baseline: dict, row: dict) -> str:
    if row.get("chamfer_squared_l2") is None:
        return "-"
    faulty = row.get("chamfer_faulty_paired_mean")
    model = row.get("chamfer_model_paired_mean")
    if faulty is None or model is None:
        return _num(row["chamfer_squared_l2"], 2)
    drop = row.get("chamfer_relative_reduction_pct")
    suffix = f" ({-drop:+.0f}%)" if drop is not None else ""
    return f"{_pair(faulty, model, 2)}{suffix}"


def _stage_table(stage_result: dict, threshold: float | None) -> list[str]:
    chosen = threshold if threshold is not None else stage_result["selected_threshold"]
    header = ["Scope", "Frames", *[label for _, label in METRICS], "Chamfer (paired)"]
    lines = [f"Threshold {chosen:.2f} "
             f"({'selected on overall ' + stage_result['selection_metric'] if threshold is None else 'user-specified'}). "
             "Cells read `faulty LiDAR -> model`.", "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for key, label in SCOPES:
        scope = stage_result["scopes"].get(key)
        if scope is None:
            continue
        row = _row_at(scope, chosen)
        base = scope["faulty_baseline"]
        cells = [label, str(row["frames"])]
        cells += [_pair(base[name], row[name]) for name, _ in METRICS]
        cells.append(_chamfer(base, row))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _full_tables(stage_result: dict) -> list[str]:
    lines = []
    for key, label in SCOPES:
        scope = stage_result["scopes"].get(key)
        if scope is None:
            continue
        base = scope["faulty_baseline"]
        lines += ["", f"**{label}** - faulty baseline: "
                  + ", ".join(f"{short} {_num(base[name])}" for name, short in METRICS),
                  "", "| Threshold | " + " | ".join(s for _, s in METRICS) + " | Chamfer |",
                  "|---|" + "---|" * (len(METRICS) + 1)]
        for row in scope["rows"]:
            lines.append(f"| {row['threshold']:.2f} | "
                         + " | ".join(_num(row[name]) for name, _ in METRICS)
                         + f" | {_num(row.get('chamfer_squared_l2'), 2)} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path,
                        help="Folder containing threshold_metrics.json")
    parser.add_argument("--threshold", type=float,
                        help="Report this threshold instead of the selected one")
    parser.add_argument("--full", action="store_true",
                        help="Also list every threshold for every fault")
    args = parser.parse_args()
    result = json.loads((args.sweep_root / "threshold_metrics.json")
                        .read_text(encoding="utf-8"))
    stages = result["stages"]
    frame_count = len(result["run_config"]["frame_ids"])
    lines = [f"# Completion sweep: {args.sweep_root.name}", "",
             f"Frames: {frame_count}; selection metric: {result['selection_metric']}."]
    missing = [name for name, _ in STAGES if name not in stages]
    for name, title in STAGES:
        if name not in stages:
            continue
        lines += ["", f"## {title}", ""]
        lines += _stage_table(stages[name], args.threshold)
        if args.full:
            lines += _full_tables(stages[name])
    if missing:
        lines += ["", f"_Not in this run: {', '.join(missing)} "
                  "(rerun the sweep with the current code)._"]
    text = "\n".join(lines) + "\n"
    (args.sweep_root / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
