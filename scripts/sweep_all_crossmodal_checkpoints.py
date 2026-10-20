"""Sweep every best radar-only occupancy checkpoint and rank comparable IoU.

Search roots should contain training runs, not the VoD dataset. Unsupported or
incompatible checkpoints are recorded as failures rather than silently ranked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


METRICS = ("iou_known_supported", "iou_full_support", "iou_exact", "iou_at_0_2m")
# These checkpoints can contain the two radar-only occupancy students accepted
# by sweep_meta_occupancy_thresholds. Encoders, teachers, and surface models do
# not emit the same voxel probabilities and cannot join this leaderboard.
STUDENT_BEST_NAMES = {"best_student.ckpt", "best_iou.ckpt",
                      "best_radar_iou.ckpt", "best_distillation.ckpt"}
CSV_COLUMNS = ("checkpoint", "model_type", "epoch", "frames", "known_iou",
               "known_threshold", "full_iou", "full_threshold", "exact_iou",
               "exact_threshold", "tolerant_iou", "tolerant_threshold", "sweep_json")


def _discover(roots: list[Path], explicit: list[Path], include_smoke: bool) -> list[Path]:
    paths = set(explicit)
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Checkpoint search root missing: {root}")
        for path in root.rglob("best_*.ckpt"):
            if path.name not in STUDENT_BEST_NAMES:
                continue
            if not include_smoke and "smoke" in path.relative_to(root).parts:
                continue
            paths.add(path)
    paths = sorted(path.resolve() for path in paths)
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Best checkpoint missing or empty: {path}")
    return paths


def _folder(path: Path) -> str:
    label = re.sub(r"[^a-zA-Z0-9_-]+", "_", "_".join(path.parts[-3:]))[:90]
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{label}_{digest}"


def _best(rows: list[dict], metric: str) -> dict:
    valid = [row for row in rows if row.get(metric) is not None]
    if not valid:
        raise ValueError(f"No defined values for {metric}")
    return max(valid, key=lambda row: (row[metric], row["threshold"]))


def _summary(path: Path, result_path: Path, expected_split: str,
             expected_frames: int | None, expected_thresholds: list[float]) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (Path(result["checkpoint"]).resolve() != path or result["split"] != expected_split
            or (expected_frames is not None and result["frames"] != expected_frames)
            or [row["threshold"] for row in result["rows"]] != expected_thresholds):
        raise ValueError(f"Sweep result identity or thresholds differ: {result_path}")
    best = {metric: _best(result["rows"], metric) for metric in METRICS}
    return {
        "checkpoint": str(path), "model_type": result["model_type"],
        "epoch": result["checkpoint_epoch"], "frames": result["frames"],
        "radar_variant": result["radar_variant"], "radius_m": result["radius_m"],
        "sweep_json": str(result_path),
        "known_iou": best["iou_known_supported"]["iou_known_supported"],
        "known_threshold": best["iou_known_supported"]["threshold"],
        "full_iou": best["iou_full_support"]["iou_full_support"],
        "full_threshold": best["iou_full_support"]["threshold"],
        "exact_iou": best["iou_exact"]["iou_exact"],
        "exact_threshold": best["iou_exact"]["threshold"],
        "tolerant_iou": best["iou_at_0_2m"]["iou_at_0_2m"],
        "tolerant_threshold": best["iou_at_0_2m"]["threshold"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--meta-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[round(index / 10, 2) for index in range(1, 10)])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    if not args.search_root and not args.checkpoint:
        parser.error("Provide at least one --search-root or --checkpoint")
    if (args.batch_size < 1 or args.num_workers < 0 or not args.thresholds
            or any(not 0 < value < 1 for value in args.thresholds)
            or len(set(args.thresholds)) != len(args.thresholds)):
        parser.error("Invalid batch size, worker count, or thresholds")
    if not (args.meta_cache_root / args.split / "manifest.json").is_file():
        parser.error("Version-3 geometry cache manifest is missing")
    paths = _discover(args.search_root, args.checkpoint, args.include_smoke)
    if not paths:
        parser.error("No best_*.ckpt files found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    thresholds = sorted(args.thresholds)
    successes, failures = [], []
    expected_frames = None
    expected_geometry = None
    for index, path in enumerate(paths, 1):
        folder = args.output_root / _folder(path)
        result_path = folder / "threshold_sweep.json"
        print(f"\n[{index}/{len(paths)}] {path}", flush=True)
        command = [
            sys.executable, "-u", "-m", "scripts.sweep_meta_occupancy_thresholds",
            "--vod-root", str(args.vod_root),
            "--data-cache-root", str(args.data_cache_root),
            "--meta-cache-root", str(args.meta_cache_root),
            "--checkpoint", str(path), "--output-root", str(folder),
            "--split", args.split, "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers), "--device", args.device,
            "--thresholds", *(str(value) for value in thresholds),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failures.append({"checkpoint": str(path), "reason": f"sweep exited {completed.returncode}"})
            continue
        try:
            summary = _summary(path, result_path, args.split, expected_frames, thresholds)
            geometry = (summary["radar_variant"], summary["radius_m"], summary["frames"])
            if expected_geometry is not None and geometry != expected_geometry:
                raise ValueError("Validation split, radar variant, or support radius differs")
            expected_geometry = geometry
            expected_frames = summary["frames"]
            successes.append(summary)
        except (KeyError, ValueError, FileNotFoundError) as error:
            failures.append({"checkpoint": str(path), "reason": str(error)})
    report = {"split": args.split, "thresholds": thresholds,
              "discovered": len(paths), "evaluated": successes, "failed": failures}
    report_path = args.output_root / "leaderboard.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output_root / "leaderboard.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({name: row[name] for name in CSV_COLUMNS} for row in successes)
    for metric, title in (("known_iou", "Known IoU"), ("full_iou", "Full IoU")):
        if successes:
            highest = max(row[metric] for row in successes)
            print(f"Best {title}: {highest:.6f}", flush=True)
            for winner in successes:
                if abs(winner[metric] - highest) <= 1e-12:
                    threshold = winner[metric.split("_")[0] + "_threshold"]
                    print(f"  threshold {threshold:.2f} | epoch {winner['epoch']} "
                          f"| {winner['checkpoint']}", flush=True)
    print(f"Evaluated {len(successes)}/{len(paths)}; report: {report_path}", flush=True)
    if failures:
        print("Unranked checkpoints:", flush=True)
        for failure in failures:
            print(f"  {failure['checkpoint']}: {failure['reason']}", flush=True)
        raise SystemExit(2)
    if not successes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
