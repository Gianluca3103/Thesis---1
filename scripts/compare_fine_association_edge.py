"""Compare raw 32-slot Stage-I probability maps with edge-aware interpolation.

This is an inference-only postprocessing experiment. Clean LiDAR is used for
validation metrics and display, never for interpolation or candidate choice.
Interpolated voxel centers must remain within 3 m of an active radar voxel.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import random

import numpy as np
from scipy.spatial import cKDTree
import torch

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage1.radar_fov import (
    RADAR_FOV_SPEC, RADAR_HALF_FOV_DEG, RADAR_MAX_RANGE_M)
from models.radar_lidar_stage1.sparse import voxelize
from models.radar_lidar_crossmodal_v1.edge_aware_surface import edge_aware_surface
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.fine_association import load_fine_association
from models.radar_lidar_crossmodal_v1.meta_occupancy import (
    scan_overlap_counts_observed, scan_overlap_rates)
from models.radar_lidar_crossmodal_v1.pointer_voxels import merge_surface_pointers
from scripts.compare_crossmodal_surface import _write_html
from scripts.visualize_crossmodal_student import _write_ply


SCOPES = ("radar_1m", "radar_3m", "full_roi")
VARIANTS = ("raw", "edge_same_threshold", "edge_matched_raw_count")
DEFAULT_THRESHOLDS = (0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                      0.95, 0.975, 0.99)
COUNT_FIELDS = ("predicted", "observed", "tp_exact", "tp_at_0_2m")
RATE_FIELDS = ("precision_exact", "recall_exact", "iou_exact", "f1_exact",
               "precision_at_0_2m", "recall_at_0_2m", "iou_at_0_2m",
               "f1_at_0_2m")


def _near_radar_voxel(xyz: np.ndarray, anchors: np.ndarray,
                      radius_m: float) -> np.ndarray:
    if not len(xyz) or not len(anchors):
        return np.zeros(len(xyz), dtype=np.bool_)
    distance, _ = cKDTree(anchors).query(xyz, k=1)
    return distance <= radius_m + 1e-6


def _rows(xyz_indices: np.ndarray) -> torch.Tensor:
    indices = torch.as_tensor(xyz_indices, dtype=torch.long)
    return torch.cat((torch.zeros((len(indices), 1), dtype=torch.long),
                      indices[:, [2, 1, 0]]), dim=1)


def _metrics(coords: torch.Tensor, scores: np.ndarray, clean_coords: torch.Tensor,
             threshold: float, *, selected: np.ndarray | None = None) -> dict:
    if selected is None:
        chosen = coords[torch.as_tensor(scores >= threshold)]
    else:
        chosen = coords[torch.as_tensor(selected, dtype=torch.long)]
    counts = scan_overlap_counts_observed(chosen, clean_coords)
    rates = scan_overlap_rates(counts)
    return {**counts, **rates,
            "precision_exact": (counts["tp_exact"] / counts["predicted"]
                                if counts["predicted"] else None),
            "recall_exact": (counts["tp_exact"] / counts["observed"]
                             if counts["observed"] else None),
            "precision_at_0_2m": (counts["tp_at_0_2m"] / counts["predicted"]
                                   if counts["predicted"] else None),
            "recall_at_0_2m": (counts["tp_at_0_2m"] / counts["observed"]
                                if counts["observed"] else None)}


def _aggregate(rows: list[dict]) -> dict:
    total = {key: sum(int(row[key]) for row in rows) for key in COUNT_FIELDS}
    rates = scan_overlap_rates(total)
    return {**total, **rates,
            "precision_exact": (total["tp_exact"] / total["predicted"]
                                if total["predicted"] else None),
            "recall_exact": (total["tp_exact"] / total["observed"]
                             if total["observed"] else None),
            "precision_at_0_2m": (total["tp_at_0_2m"] / total["predicted"]
                                   if total["predicted"] else None),
            "recall_at_0_2m": (total["tp_at_0_2m"] / total["observed"]
                                if total["observed"] else None)}


def _scope_masks(raw_xyz: np.ndarray, edge_xyz: np.ndarray,
                 clean_xyz: np.ndarray, anchors: np.ndarray) -> dict:
    masks = {}
    for label, radius_m in (("radar_1m", 1.0), ("radar_3m", 3.0)):
        masks[label] = {
            "raw": _near_radar_voxel(raw_xyz, anchors, radius_m),
            "edge": _near_radar_voxel(edge_xyz, anchors, radius_m),
            "clean": _near_radar_voxel(clean_xyz, anchors, radius_m),
        }
    masks["full_roi"] = {
        "raw": np.ones(len(raw_xyz), dtype=np.bool_),
        "edge": np.ones(len(edge_xyz), dtype=np.bool_),
        "clean": np.ones(len(clean_xyz), dtype=np.bool_),
    }
    return masks


def _sweep_frame(raw_coords: torch.Tensor, raw_score: np.ndarray,
                 edge_coords: torch.Tensor, edge_score: np.ndarray,
                 clean_coords: torch.Tensor, masks: dict,
                 thresholds: tuple[float, ...]) -> dict:
    """Score each radius with its own clean denominator and matched cell count."""
    results = {}
    for scope in SCOPES:
        mask = masks[scope]
        clean = clean_coords[torch.from_numpy(mask["clean"])]
        raw_scope = raw_coords[torch.from_numpy(mask["raw"])]
        edge_scope = edge_coords[torch.from_numpy(mask["edge"])]
        raw_scores = raw_score[mask["raw"]]
        edge_scores = edge_score[mask["edge"]]
        order = np.argsort(-edge_scores, kind="stable")
        rows = {}
        for threshold in thresholds:
            raw = _metrics(raw_scope, raw_scores, clean, threshold)
            edge = _metrics(edge_scope, edge_scores, clean, threshold)
            matched = _metrics(edge_scope, edge_scores, clean, threshold,
                               selected=order[:raw["predicted"]])
            rows[threshold] = {"raw": raw, "edge_same_threshold": edge,
                               "edge_matched_raw_count": matched}
        results[scope] = rows
    return results


def _new_sweep_totals(thresholds: tuple[float, ...]) -> dict:
    return {(scope, threshold, variant): {name: 0 for name in COUNT_FIELDS}
            for scope in SCOPES for threshold in thresholds
            for variant in VARIANTS}


def _add_sweep(totals: dict, frame: dict) -> None:
    for scope in SCOPES:
        for threshold, variants in frame[scope].items():
            for variant in VARIANTS:
                target = totals[(scope, threshold, variant)]
                source = variants[variant]
                for name in COUNT_FIELDS:
                    target[name] += source[name]


def _aggregate_sweep(totals: dict,
                     thresholds: tuple[float, ...]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        for scope in SCOPES:
            for variant in VARIANTS:
                aggregate = _aggregate([totals[(scope, threshold, variant)]])
                rows.append({"threshold": threshold, "scope": scope,
                             "variant": variant, **aggregate})
    return rows


def _best_thresholds(rows: list[dict]) -> dict:
    best = {}
    for scope in SCOPES:
        best[scope] = {}
        for variant in VARIANTS:
            candidates = [row for row in rows if row["scope"] == scope
                          and row["variant"] == variant]
            best[scope][variant] = {
                metric: (max(candidates, key=lambda row: (
                    -1 if row[metric] is None else row[metric],
                    row["threshold"]))["threshold"])
                for metric in RATE_FIELDS}
    return best


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--frame-id", nargs="+")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=3103)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=DEFAULT_THRESHOLDS,
                        help="Confidence thresholds for the regional metric sweep")
    parser.add_argument("--edge-seed-threshold", type=float, default=0.25)
    parser.add_argument("--max-link-m", type=float, default=2.0)
    parser.add_argument("--min-affinity", type=float, default=0.20)
    parser.add_argument("--lattice-stride", type=int, default=2)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.limit < 1 or not 0 < args.threshold < 1
            or not args.thresholds or any(not 0 < t < 1 for t in args.thresholds)
            or len(set(args.thresholds)) != len(args.thresholds)
            or not 0 <= args.edge_seed_threshold <= 1
            or not 0 < args.min_affinity < 1 or args.max_link_m <= 0
            or args.lattice_stride < 1):
        parser.error("Invalid frame limit, threshold, or interpolation setting")
    thresholds = tuple(sorted(set((*args.thresholds, args.threshold))))
    if args.edge_seed_threshold > thresholds[0]:
        parser.error("--edge-seed-threshold must be no greater than the lowest sweep threshold")
    student, saved = load_fine_association(args.checkpoint, args.device)
    if saved.get("lidar_fov") != RADAR_FOV_SPEC:
        raise ValueError("Checkpoint and clean-LiDAR validation sector differ")
    if saved["config"]["proposals_per_anchor"] != 32:
        raise ValueError("Expected the 32-candidate fine-association checkpoint")
    radius_m = float(saved["radius_m"])
    dataset = VoDStage1Dataset(
        args.vod_root, args.split, radar_variant=saved["radar_variant"],
        frame_ids=args.frame_id, include_clean=True,
        lidar_radar_fov_deg=RADAR_HALF_FOV_DEG,
        lidar_radar_max_range_m=RADAR_MAX_RANGE_M)
    if args.frame_id:
        found = {frame.frame_id for frame in dataset.frames}
        if missing := set(args.frame_id) - found:
            raise ValueError(f"Frames absent from {args.split}: {sorted(missing)}")
    else:
        dataset.frames = random.Random(args.random_seed).sample(
            dataset.frames, min(args.limit, len(dataset.frames)))
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    sweep_totals = _new_sweep_totals(thresholds)
    for row in dataset:
        batch = collate_stage1([row])
        output = student(batch["radar"].to(args.device),
                         batch["radar_valid"].to(args.device))
        surface = output.surface
        field = merge_surface_pointers(surface)
        raw_coords = field.coords.cpu()
        raw_xyz = GRID.centers_xyz(raw_coords).numpy()
        raw_score = field.confidence.cpu().numpy()
        anchors = surface.anchor_xyz.cpu().numpy()
        guide = surface.guide[field.representative_source].cpu().numpy()
        groups = field.representative_source.cpu().numpy()
        edge = edge_aware_surface(
            raw_xyz, raw_score, guide, groups,
            seed_threshold=args.edge_seed_threshold,
            lattice_stride=args.lattice_stride, max_link_m=args.max_link_m,
            min_affinity=args.min_affinity)
        edge_xyz_all = edge.centers_xyz()
        supported = _near_radar_voxel(edge_xyz_all, anchors, radius_m)
        edge_xyz = edge_xyz_all[supported]
        edge_score = edge.scores[supported]
        edge_coords = _rows(edge.coordinates_xyz[supported])

        clean = row["clean_lidar"][:, :3].numpy()
        clean_points = row["clean_lidar"][None]
        clean_valid = torch.ones((1, len(clean)), dtype=torch.bool)
        clean_coords, _, _, _ = voxelize(clean_points, clean_valid, GRID)
        clean_centers = GRID.centers_xyz(clean_coords).numpy()
        masks = _scope_masks(raw_xyz, edge_xyz, clean_centers, anchors)
        sweep = _sweep_frame(raw_coords, raw_score, edge_coords, edge_score,
                             clean_coords, masks, thresholds)
        _add_sweep(sweep_totals, sweep)
        preview = sweep["radar_3m"][args.threshold]
        report = {"frame_id": row["frame_id"],
                  "raw": preview["raw"],
                  "edge_same_threshold": preview["edge_same_threshold"],
                  "edge_matched_raw_count": preview["edge_matched_raw_count"],
                  "edge_accepted_links": edge.accepted_edges,
                  "edge_outside_3m_removed": int((~supported).sum()),
                  "raw_total_candidates": len(raw_xyz),
                  "edge_total_candidates": len(edge_xyz)}
        reports.append(report)
        if not args.metrics_only:
            stem = args.output_root / f"{row['frame_id']}_edge_comparison"
            _write_ply(stem.with_name(stem.name + "_raw.ply"), raw_xyz, raw_score)
            _write_ply(stem.with_name(stem.name + "_edge.ply"), edge_xyz, edge_score)
            viewer = stem.with_suffix(".html")
            _write_html(viewer, row["frame_id"], saved["epoch"],
                        row["radar"][:, :3].numpy(), clean,
                        raw_xyz, raw_score, edge_xyz, edge_score,
                        threshold=args.threshold, links=edge.accepted_edges)
            report["viewer"] = viewer.name
        print(f"{row['frame_id']}: 1 m raw P/R "
              f"{sweep['radar_1m'][args.threshold]['raw']['precision_exact']}/"
              f"{sweep['radar_1m'][args.threshold]['raw']['recall_exact']}; "
              f"3 m raw P/R {preview['raw']['precision_exact']}/"
              f"{preview['raw']['recall_exact']}", flush=True)

    if not reports:
        raise RuntimeError("The selected split has no frames to compare")
    sweep_rows = _aggregate_sweep(sweep_totals, thresholds)
    summary = {"checkpoint": str(args.checkpoint), "epoch": saved["epoch"],
               "split": args.split, "frames": len(reports),
               "radius_m": radius_m, "slots_per_radar_voxel": 32,
               "threshold": args.threshold, "sweep_thresholds": thresholds,
               "edge_parameters": {"seed_threshold": args.edge_seed_threshold,
                                   "max_link_m": args.max_link_m,
                                   "min_affinity": args.min_affinity,
                                   "lattice_stride": args.lattice_stride},
               "metric_scope": {
                   "radar_1m": "Predicted and all clean voxels with centers within 1 m of an active fine radar voxel center",
                   "radar_3m": "Predicted and all clean voxels with centers within 3 m of an active fine radar voxel center",
                   "full_roi": "All predicted and clean voxels in the forward-FOV model grid; areas without predictions count as misses",
               },
               "metric_note": "Voxel scan agreement, not proof that unobserved space is empty; 0.2 m matches are one-to-one within each scope",
               "sweep": sweep_rows,
               "best_thresholds": _best_thresholds(sweep_rows),
               "raw": _aggregate([item["raw"] for item in reports]),
               "edge_same_threshold": _aggregate(
                   [item["edge_same_threshold"] for item in reports]),
               "edge_matched_raw_count": _aggregate(
                   [item["edge_matched_raw_count"] for item in reports]),
               "per_frame": reports}
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = args.output_root / "threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        columns = ("threshold", "scope", "variant", *COUNT_FIELDS, *RATE_FIELDS)
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: item[key] for key in columns}
                         for item in sweep_rows)
    print("\nValidation scan agreement (same frames; exact voxels):", flush=True)
    print("variant                 scope      threshold   precision     recall        IoU         F1", flush=True)
    for scope in SCOPES:
        for variant in VARIANTS:
            threshold = summary["best_thresholds"][scope][variant]["f1_exact"]
            values = next(item for item in sweep_rows if item["scope"] == scope
                          and item["variant"] == variant
                          and item["threshold"] == threshold)
            def rate(name):
                value = values[name]
                return f"{value:.4f}" if value is not None else "n/a"
            print(f"{variant:<23} {scope:<10} {threshold:>9.3f}  "
                  f"{rate('precision_exact'):>10}  {rate('recall_exact'):>9}  "
                  f"{rate('iou_exact'):>9}  {rate('f1_exact'):>9}", flush=True)
    if not args.metrics_only and reports:
        links = "".join(
            f'<li><a target="viewer" href="{html.escape(item["viewer"])}">'
            f'Frame {html.escape(item["frame_id"])}</a></li>'
            for item in reports)
        first = html.escape(reports[0]["viewer"])
        index = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
                 "<title>Edge-aware Stage I comparison</title>"
                 "<style>body{margin:1rem;background:#101820;color:#edf4fa;"
                 "font:16px system-ui}a{color:#88ceff}"
                 "iframe{width:100%;height:85vh;border:1px solid #526474}"
                 "</style><h1>Raw versus edge-aware Stage I candidates</h1>"
                 f"<ul>{links}</ul><iframe name=\"viewer\" src=\"{first}\" "
                 "title=\"Edge-aware comparison\"></iframe></html>")
        (args.output_root / "index.html").write_text(index, encoding="utf-8")
    print(f"Summary: {args.output_root / 'summary.json'}", flush=True)
    print(f"Sweep: {csv_path}", flush=True)
    if not args.metrics_only:
        print(f"Open: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
