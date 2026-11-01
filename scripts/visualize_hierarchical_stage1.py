"""Interactive Stage-I gate views beside radar, faulty LiDAR, and clean targets.

The colored gate points are candidate voxel centers, not reconstructed LiDAR
returns. Clean LiDAR is used only for reference and exact-match audit colors.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import random

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.hierarchical_gate import (
    fine_ranking, load_hierarchical_gate, missing_target, target_mask)
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.fault_data import SevereFogCache
from models.radar_lidar_stage2.sparse_geometry import GRID
from scripts.compare_crossmodal_surface import _write_html
from scripts.train_hierarchical_stage1 import faulted
from scripts.visualize_crossmodal_student import _write_ply


def _centers(coords: torch.Tensor) -> np.ndarray:
    if not len(coords):
        return np.empty((0, 3), dtype=np.float32)
    xyz = coords[:, [2, 1, 0]].float() + 0.5
    xyz = xyz * xyz.new_tensor(GRID.voxel_xyz) + xyz.new_tensor(GRID.minimum_xyz)
    return xyz.cpu().numpy()


def _roi(points: torch.Tensor) -> np.ndarray:
    xyz = points[:, :3].cpu().numpy()
    low = np.asarray(GRID.minimum_xyz)
    high = np.asarray(GRID.maximum_xyz)
    valid = np.isfinite(xyz).all(axis=1) & ((xyz >= low) & (xyz < high)).all(axis=1)
    return xyz[valid]


def _frame_ids(dataset, saved: dict, args) -> list[str]:
    known = {frame.frame_id for frame in dataset.frames}
    if args.frame_id:
        missing = set(args.frame_id) - known
        if missing:
            raise ValueError(f"Frame IDs absent from validation split: {sorted(missing)}")
        return list(dict.fromkeys(args.frame_id))[:args.limit]
    if args.comparison:
        report = json.loads(args.comparison.read_text(encoding="utf-8"))
        if report.get("hierarchical_sha256") != hashlib.sha256(
                args.checkpoint.read_bytes()).hexdigest():
            raise ValueError("Comparison JSON belongs to a different checkpoint")
        if "sampled_frame_ids" not in report:
            raise ValueError("Comparison predates frame-ID recording; use --frame-id")
        ids = report["sampled_frame_ids"][report["calibration_frames"]:]
        if not ids or any(frame_id not in known for frame_id in ids):
            raise ValueError("Comparison has no valid held-out frame IDs")
        return ids[:args.limit]
    excluded = (set(saved.get("selection_frame_ids", ())) |
                set(saved.get("inherited_selection_frame_ids", ())))
    if args.temporal_gap and excluded:
        excluded_numeric = tuple(int(frame_id) for frame_id in excluded)
        candidates = [frame.frame_id for frame in dataset.frames if all(
            abs(int(frame.frame_id) - prior) > args.temporal_gap
            for prior in excluded_numeric)]
    else:
        candidates = [frame.frame_id for frame in dataset.frames
                      if frame.frame_id not in excluded]
    selected: list[str] = []
    for frame_id in random.Random(args.seed).sample(candidates, len(candidates)):
        if (not args.temporal_gap or all(
                abs(int(frame_id) - int(prior)) > args.temporal_gap
                for prior in selected)):
            selected.append(frame_id)
            if len(selected) == args.limit:
                break
    if not selected:
        raise ValueError("No eligible validation frames remain")
    return selected


def _index(path: Path, reports: list[dict]) -> None:
    links = "".join(
        f'<li><a target="viewer" href="{html.escape(row["viewer"])}">'
        f'Frame {html.escape(row["frame_id"])} · {html.escape(row["fault"])}'
        f' · P {row["precision"]:.1%} · R {row["recall"]:.1%}</a></li>'
        for row in reports)
    first = html.escape(reports[0]["viewer"])
    markup = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
              "<title>Hierarchical Stage I gate views</title>"
              "<style>body{margin:1rem;background:#101820;color:#edf4fa;"
              "font:16px system-ui}a{color:#88ceff}"
              "iframe{width:100%;height:85vh;border:1px solid #526474}"
              "</style><h1>Hierarchical Stage I candidate gate</h1>"
              "<p>Gate points are voxel centers proposed for Stage II, not "
              "reconstructed LiDAR returns.</p>"
              f"<ul>{links}</ul><iframe name='viewer' src='{first}' "
              "title='Stage I gate comparison'></iframe></html>")
    path.write_text(markup, encoding="utf-8")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--fog-cache-root", type=Path)
    parser.add_argument("--comparison", type=Path,
                        help="Use held-out IDs from a matched comparison JSON")
    parser.add_argument("--frame-id", nargs="+", help="Specific validation frame IDs")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--temporal-gap", type=int, default=20)
    parser.add_argument("--score-mode", choices=("hierarchical", "fine_only"),
                        default="hierarchical")
    parser.add_argument("--fault-modes", nargs="+",
                        choices=("total_loss", "fov_filter", "fog_sim"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit < 1 or args.budget < 1 or args.temporal_gap < 0:
        parser.error("Limit and budget must be positive; temporal gap cannot be negative")
    model, saved = load_hierarchical_gate(args.checkpoint, args.device)
    modes = tuple(args.fault_modes or saved["fault_modes"])
    if "fog_sim" in modes and args.fog_cache_root is None:
        parser.error("fog_sim requires --fog-cache-root")
    fog = (SevereFogCache(args.fog_cache_root, "val", None,
                          seed=saved["seed"], check_complete=False)
           if args.fog_cache_root else None)
    dataset = paired_dataset(args.vod_root, "val", saved["radar_variant"],
                             data_cache_root=args.data_cache_root)
    frame_ids = _frame_ids(dataset, saved, args)
    by_id = {frame.frame_id: index for index, frame in enumerate(dataset.frames)}
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for frame_id in frame_ids:
        row = dataset[by_id[frame_id]]
        radar = row["radar"].to(args.device)
        clean = row["clean_lidar"]
        for mode in modes:
            faulty = faulted(clean, mode, frame_id=frame_id, fog_cache=fog)
            faulty_device = faulty.to(args.device)
            target = missing_target(clean[:, :3].to(args.device), faulty_device)
            output = model(radar, faulty_device)
            ranking, probabilities = fine_ranking(output, args.score_mode)
            chosen = torch.topk(ranking, min(args.budget, len(ranking))).indices
            selected = output["fine_coords"][chosen]
            matched_selected = target_mask(selected, target, 1)
            matched_target = target_mask(target, selected, 1)
            tp = int(matched_selected.sum())
            assert tp == int(matched_target.sum())
            pool_tp = int(target_mask(output["fine_coords"], target, 1).sum())
            predicted_xyz = _centers(selected)
            target_xyz = _centers(target)
            display_score = probabilities[chosen].cpu().numpy()
            audit_xyz = np.concatenate((target_xyz[~matched_target.cpu().numpy()],
                                        predicted_xyz))
            audit_color = (["#67aaff"] * int((~matched_target).sum()) +
                           ["#41d17d" if hit else "#ff6868"
                            for hit in matched_selected.cpu().tolist()])
            precision = tp / len(selected) if len(selected) else 0.0
            recall = tp / len(target) if len(target) else 1.0
            stem = f"{frame_id}_{mode}_top{args.budget}"
            viewer = args.output_root / f"{stem}.html"
            _write_html(
                viewer, frame_id, int(saved["epoch"]), _roi(row["radar"]),
                _roi(clean), predicted_xyz, display_score, target_xyz,
                np.ones(len(target_xyz), dtype=np.float32), threshold=0.0,
                links=None,
                audit_clouds=[
                    {"name": "Surviving faulty LiDAR", "xyz": np.round(
                        _roi(faulty), 3).tolist(),
                     "color": ["#ffab40"] * len(_roi(faulty))},
                    {"name": "Exact gate audit", "xyz": np.round(
                        audit_xyz, 3).tolist(), "color": audit_color,
                     "summary": f"TP {tp:,} · FP {len(selected)-tp:,} · FN {len(target)-tp:,}"},
                ],
                raw_label=f"Selected Stage-II candidates (top {args.budget:,})",
                edge_label="Missing clean target voxels",
                page_title=f"Hierarchical Stage I · {mode}",
                description=("The gate shows proposed voxel centers, not a "
                             "reconstructed LiDAR scan. Clean LiDAR is reference only. "
                             f"Selected precision {precision:.1%}; recall {recall:.1%}; "
                             f"candidate-pool oracle recall "
                             f"{pool_tp / len(target) if len(target) else 1.0:.1%}."),
                audit_note=("Exact voxel audit: green selected target, red "
                            "selected non-target, blue missing target. "
                            "Drag to rotate; Shift-drag to pan; wheel to zoom."))
            _write_ply(args.output_root / f"{stem}_gate.ply",
                       predicted_xyz, display_score)
            _write_ply(args.output_root / f"{stem}_missing_clean.ply", target_xyz)
            reports.append({"frame_id": frame_id, "fault": mode,
                            "budget": args.budget, "selected": len(selected),
                            "target": len(target), "tp": tp,
                            "precision": precision, "recall": recall,
                            "pool_sites": len(output["fine_coords"]),
                            "pool_tp": pool_tp,
                            "pool_recall": (pool_tp / len(target)
                                            if len(target) else 1.0),
                            "viewer": viewer.name})
    summary = {"checkpoint": str(args.checkpoint.resolve()),
               "epoch": int(saved["epoch"]), "score_mode": args.score_mode,
               "frame_ids": frame_ids, "reports": reports}
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    _index(args.output_root / "index.html", reports)
    print(f"Open: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
