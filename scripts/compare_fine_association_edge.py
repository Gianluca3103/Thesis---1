"""Compare raw 32-slot Stage-I probability maps with edge-aware interpolation.

This is an inference-only postprocessing experiment. Clean LiDAR is used for
validation metrics and display, never for interpolation or candidate choice.
Interpolated voxel centers must remain within 3 m of an active radar voxel.
"""

from __future__ import annotations

import argparse
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
    total = {key: sum(int(row[key]) for row in rows)
             for key in ("predicted", "observed", "tp_exact", "tp_at_0_2m")}
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
    parser.add_argument("--edge-seed-threshold", type=float, default=0.25)
    parser.add_argument("--max-link-m", type=float, default=2.0)
    parser.add_argument("--min-affinity", type=float, default=0.20)
    parser.add_argument("--lattice-stride", type=int, default=2)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.limit < 1 or not 0 < args.threshold < 1
            or not 0 <= args.edge_seed_threshold <= 1
            or not 0 < args.min_affinity < 1 or args.max_link_m <= 0
            or args.lattice_stride < 1):
        parser.error("Invalid frame limit, threshold, or interpolation setting")
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
        clean_coords = clean_coords[
            torch.as_tensor(_near_radar_voxel(clean_centers, anchors, radius_m))]
        raw_result = _metrics(raw_coords, raw_score, clean_coords, args.threshold)
        edge_result = _metrics(edge_coords, edge_score, clean_coords, args.threshold)
        raw_count = raw_result["predicted"]
        matched_indices = np.argsort(-edge_score, kind="stable")[:raw_count]
        matched_result = _metrics(edge_coords, edge_score, clean_coords,
                                  args.threshold, selected=matched_indices)
        report = {"frame_id": row["frame_id"], "raw": raw_result,
                  "edge_same_threshold": edge_result,
                  "edge_matched_raw_count": matched_result,
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
        print(f"{row['frame_id']}: raw IoU {raw_result['iou_exact']}, "
              f"edge IoU {edge_result['iou_exact']}, "
              f"edge equal-count IoU {matched_result['iou_exact']}", flush=True)

    if not reports:
        raise RuntimeError("The selected split has no frames to compare")
    summary = {"checkpoint": str(args.checkpoint), "epoch": saved["epoch"],
               "split": args.split, "frames": len(reports),
               "radius_m": radius_m, "slots_per_radar_voxel": 32,
               "threshold": args.threshold,
               "edge_parameters": {"seed_threshold": args.edge_seed_threshold,
                                   "max_link_m": args.max_link_m,
                                   "min_affinity": args.min_affinity,
                                   "lattice_stride": args.lattice_stride},
               "metric_scope": "Clean LiDAR scan voxels within 3 m of active radar voxels",
               "raw": _aggregate([item["raw"] for item in reports]),
               "edge_same_threshold": _aggregate(
                   [item["edge_same_threshold"] for item in reports]),
               "edge_matched_raw_count": _aggregate(
                   [item["edge_matched_raw_count"] for item in reports]),
               "per_frame": reports}
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\nValidation scan agreement (same frames and 3 m radar support):", flush=True)
    print("variant                    cells     IoU exact  IoU@0.2m  F1 exact  F1@0.2m", flush=True)
    for label, key in (("Raw", "raw"),
                       ("Edge, same threshold", "edge_same_threshold"),
                       ("Edge, matched count", "edge_matched_raw_count")):
        values = summary[key]
        def rate(name):
            value = values[name]
            return f"{value:.4f}" if value is not None else "n/a"
        print(f"{label:<25} {values['predicted']:>8}  "
              f"{rate('iou_exact'):>9}  {rate('iou_at_0_2m'):>9}  "
              f"{rate('f1_exact'):>8}  {rate('f1_at_0_2m'):>8}", flush=True)
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
    if not args.metrics_only:
        print(f"Open: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
