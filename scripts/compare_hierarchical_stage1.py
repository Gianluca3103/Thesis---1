"""Matched Stage-I gate evaluation on disjoint calibration and held-out frames.

Both models are scored on the same Stage-II ROI and the same fault-specific
missing clean voxels. Exact voxel matches are the primary Stage-II gate metric.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

from scipy.spatial import cKDTree
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.fine_association import load_fine_association
from models.radar_lidar_crossmodal_v1.encoder import GRID as STAGE1_GRID
from models.radar_lidar_crossmodal_v1.hierarchical_gate import (
    fine_ranking, load_hierarchical_gate, missing_target, target_mask)
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.fault_data import SevereFogCache
from models.radar_lidar_stage2.sparse_geometry import GRID as STAGE2_GRID, select_allowed
from models.radar_lidar_stage1.sparse import voxelize
from scripts.train_hierarchical_stage1 import _single, faulted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rates(count: dict[str, int]) -> dict:
    tp, selected, target = count["tp"], count["selected"], count["target"]
    return {**count,
            "precision": tp / selected if selected else 0.0,
            "recall": tp / target if target else 1.0}


def _accumulate(total: dict, coords: torch.Tensor, scores: torch.Tensor,
                target: torch.Tensor, budgets: tuple[int, ...]) -> None:
    ordered = torch.argsort(scores, descending=True, stable=True)
    for budget in (*budgets, 0):
        selected = coords if budget == 0 else coords[ordered[:budget]]
        tp = int(target_mask(selected, target, 1).sum())
        row = total["pool"] if budget == 0 else total[str(budget)]
        row["tp"] += tp
        row["selected"] += len(selected)
        row["target"] += len(target)


def _new_totals(budgets: tuple[int, ...]) -> dict:
    return {key: {"tp": 0, "selected": 0, "target": 0}
            for key in ("pool", *(str(b) for b in budgets))}


def _select_budget(rows: dict, budgets: tuple[int, ...], minimum_precision: float) -> int:
    feasible = [budget for budget in budgets
                if rows[str(budget)]["precision"] >= minimum_precision]
    if feasible:
        return max(feasible, key=lambda budget: (rows[str(budget)]["recall"],
                                                  -rows[str(budget)]["selected"]))
    return max(budgets, key=lambda budget: (
        rows[str(budget)]["recall"] * min(
            rows[str(budget)]["precision"] / minimum_precision, 1.0),
        rows[str(budget)]["recall"]))


@torch.no_grad()
def _evaluate(models: dict, loader, device: str, modes: tuple[str, ...],
              budgets: tuple[int, ...], fog_cache, score_mode: str) -> dict:
    totals = {name: {mode: _new_totals(budgets) for mode in modes}
              for name in models}
    parent_totals = {mode: {scale: {"tp": 0, "selected": 0, "target": 0}
                            for scale in (4, 2)} for mode in modes}
    radar_reach = {mode: {"within_radius": 0, "target": 0} for mode in modes}
    for row in tqdm(loader, desc="Matched Stage-I gate comparison", unit="frame"):
        radar = row["radar"].to(device)
        clean = row["clean_lidar"]
        radar_sites, _, _, _ = voxelize(
            radar[None], torch.ones((1, len(radar)), device=radar.device,
                                     dtype=torch.bool), STAGE1_GRID)
        radar_centers = STAGE1_GRID.centers_xyz(radar_sites).cpu().numpy()
        radar_tree = cKDTree(radar_centers) if len(radar_centers) else None
        baseline_field = None
        if "64_slot_baseline" in models:
            baseline = models["64_slot_baseline"]
            output = baseline(radar[None], torch.ones(
                (1, len(radar)), device=radar.device, dtype=torch.bool)).exported()
            baseline_coords, baseline_scores = select_allowed(
                output["confidence_coords"], output["confidence_probabilities"],
                1e-7)
            baseline_field = baseline_coords[:, 1:]
        for mode in modes:
            faulty = faulted(clean, mode, frame_id=row["frame_id"],
                             fog_cache=fog_cache).to(device)
            target = missing_target(clean[:, :3].to(device), faulty)
            if len(target) and radar_tree is not None:
                origin = target.new_tensor(STAGE2_GRID.minimum_xyz,
                                           dtype=torch.float32)
                size = target.new_tensor(STAGE2_GRID.voxel_xyz,
                                         dtype=torch.float32)
                target_xyz = (origin + (target[:, [2, 1, 0]].float() + 0.5) * size)
                distance, _ = radar_tree.query(target_xyz.cpu().numpy(), k=1)
                radar_reach[mode]["within_radius"] += int((distance <= 1.6 + 1e-6).sum())
            radar_reach[mode]["target"] += len(target)
            generated = models["hierarchical"](radar, faulty)
            rank, _ = fine_ranking(generated, score_mode)
            _accumulate(totals["hierarchical"][mode], generated["fine_coords"],
                        rank, target, budgets)
            for scale, generated_coords in (
                    (4, torch.unique(generated["middle_coords"] // 2, dim=0)),
                    (2, generated["middle_coords"])):
                parent_totals[mode][scale]["tp"] += int(target_mask(
                    generated_coords, target, scale).sum())
                parent_totals[mode][scale]["selected"] += len(generated_coords)
                parent_totals[mode][scale]["target"] += len(torch.unique(
                    target // scale, dim=0))
            if baseline_field is not None:
                _accumulate(totals["64_slot_baseline"][mode], baseline_field,
                            baseline_scores, target, budgets)
    result = {name: {mode: {budget: _rates(count) for budget, count in rows.items()}
                     for mode, rows in modes_rows.items()}
              for name, modes_rows in totals.items()}
    for mode in modes:
        result["hierarchical"][mode]["parent_coverage"] = {
            str(scale): _rates(parent_totals[mode][scale]) for scale in (4, 2)}
        result["hierarchical"][mode]["radar_1p6m_geometric_reach"] = {
            **radar_reach[mode],
            "recall": (radar_reach[mode]["within_radius"] / radar_reach[mode]["target"]
                       if radar_reach[mode]["target"] else 1.0)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--fog-cache-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--model-selection-val-limit", type=int, default=0,
                        help="For checkpoints made before selection IDs were stored")
    parser.add_argument("--temporal-gap", type=int, default=0,
                        help="Exclude validation frames within this many frame IDs of selection frames and each other")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--budgets", type=int, nargs="+", default=(5000, 10000, 20000, 40000))
    parser.add_argument("--minimum-precision", type=float, default=0.20)
    parser.add_argument("--score-mode", choices=("hierarchical", "fine_only"),
                        default="hierarchical")
    parser.add_argument("--coarse-parent-budget", type=int,
                        help="Inference-only parent-budget ablation")
    parser.add_argument("--middle-parent-budget", type=int,
                        help="Inference-only parent-budget ablation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.limit < 2 or args.num_workers < 0 or args.model_selection_val_limit < 0
            or args.temporal_gap < 0
            or not args.budgets
            or min(args.budgets) < 1 or not 0 < args.minimum_precision <= 1
            or (args.coarse_parent_budget is not None and args.coarse_parent_budget < 1)
            or (args.middle_parent_budget is not None and args.middle_parent_budget < 1)):
        parser.error("Invalid frame limit, worker count, budget, or precision")
    budgets = tuple(sorted(set(args.budgets)))
    hierarchical, saved = load_hierarchical_gate(args.checkpoint, args.device)
    if args.coarse_parent_budget is not None or args.middle_parent_budget is not None:
        hierarchical.config = replace(
            hierarchical.config,
            coarse_parent_budget=(args.coarse_parent_budget
                                  or hierarchical.config.coarse_parent_budget),
            middle_parent_budget=(args.middle_parent_budget
                                  or hierarchical.config.middle_parent_budget))
    models = {"hierarchical": hierarchical}
    if args.baseline_checkpoint:
        baseline, baseline_saved = load_fine_association(args.baseline_checkpoint, args.device)
        if baseline_saved["radar_variant"] != saved["radar_variant"]:
            parser.error("Both checkpoints must use the same radar variant")
        models["64_slot_baseline"] = baseline
    modes = tuple(saved["fault_modes"])
    if "fog_sim" in modes and args.fog_cache_root is None:
        parser.error("fog_sim comparison requires --fog-cache-root")
    fog = (SevereFogCache(args.fog_cache_root, "val", None,
                          seed=saved["seed"], check_complete=False)
           if args.fog_cache_root else None)
    dataset = paired_dataset(args.vod_root, "val", saved["radar_variant"])
    excluded = (set(saved.get("selection_frame_ids", ())) |
                set(saved.get("inherited_selection_frame_ids", ())))
    if not excluded and args.model_selection_val_limit:
        indices = random.Random(saved["seed"]).sample(
            range(len(dataset)), min(args.model_selection_val_limit, len(dataset)))
        excluded = {dataset.frames[index].frame_id for index in indices}
    available = [index for index, frame in enumerate(dataset.frames)
                 if frame.frame_id not in excluded]
    if args.temporal_gap:
        try:
            excluded_numeric = tuple(int(frame_id) for frame_id in excluded)
            numeric_ids = {index: int(dataset.frames[index].frame_id)
                           for index in available}
        except ValueError:
            parser.error("--temporal-gap requires numeric VoD frame IDs")
        available = [index for index in available if all(
            abs(numeric_ids[index] - frame_id) > args.temporal_gap
            for frame_id in excluded_numeric)]
        selected = []
        for index in random.Random(args.seed).sample(available, len(available)):
            if all(abs(numeric_ids[index] - numeric_ids[prior]) > args.temporal_gap
                   for prior in selected):
                selected.append(index)
                if len(selected) == args.limit:
                    break
    else:
        selected = random.Random(args.seed).sample(available,
                                                   min(args.limit, len(available)))
    half = len(selected) // 2
    if not half:
        parser.error("At least two validation frames are required")
    loader_args = dict(batch_size=1, shuffle=False, num_workers=args.num_workers,
                       collate_fn=_single)
    calibration = _evaluate(models, DataLoader(Subset(dataset, selected[:half]),
                                               **loader_args), args.device,
                            modes, budgets, fog, args.score_mode)
    heldout = _evaluate(models, DataLoader(Subset(dataset, selected[half:]),
                                           **loader_args), args.device,
                        modes, budgets, fog, args.score_mode)
    selected_budgets = {
        name: {mode: _select_budget(calibration[name][mode], budgets,
                                   args.minimum_precision) for mode in modes}
        for name in models}
    result = {
        "metric": "exact missing clean voxel within Stage-II ROI",
        "radar_variant": saved["radar_variant"],
        "hierarchical_checkpoint": str(args.checkpoint.resolve()),
        "hierarchical_sha256": _sha256(args.checkpoint),
        "inference_parent_budgets": {
            "coarse": hierarchical.config.coarse_parent_budget,
            "middle": hierarchical.config.middle_parent_budget},
        "score_mode": args.score_mode,
        "baseline_checkpoint": (str(args.baseline_checkpoint.resolve())
                                if args.baseline_checkpoint else None),
        "baseline_sha256": (_sha256(args.baseline_checkpoint)
                            if args.baseline_checkpoint else None),
        "excluded_model_selection_frames": len(excluded),
        "temporal_gap_frames": args.temporal_gap,
        "sampled_frame_ids": [dataset.frames[index].frame_id for index in selected],
        "sampled_frames": len(selected), "calibration_frames": half,
        "heldout_frames": len(selected) - half,
        "budgets": budgets, "minimum_precision": args.minimum_precision,
        "calibration": calibration, "heldout": heldout,
        "selected_budgets": selected_budgets,
        "heldout_at_calibrated_budget": {
            name: {mode: heldout[name][mode][str(selected_budgets[name][mode])]
                   for mode in modes} for name in models},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["heldout_at_calibrated_budget"], indent=2), flush=True)
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
