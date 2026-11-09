"""Render balanced 3D examples from a deployable conditional Stage II checkpoint.

Prediction uses Gaussian noise, faulty LiDAR, and radar. Clean LiDAR is loaded
only after prediction to color additions and report reconstruction metrics.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.clean_selected_metrics import (
    add_frame, completion_summary, empty_accumulator, full_frame_voxels,
    voxel_centers)
from models.radar_lidar_stage2.clean_thresholds import THRESHOLD_STEPS
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import cosine_alpha_bars
from scripts.conditional_stage2 import (
    deployment_loader, load_model, predict_batch, reference_clean)
from scripts.evaluate_stage2_conditional_reconstruction import _frame_candidates
from scripts.train_stage2_clean_denoiser import _rows
from scripts.visualize_stage2_conditioned_completion import (
    _roi_radar, _save_record, balanced_frame_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--fault-region-root", type=Path, required=True)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--per-fault", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available() or
            not 0 <= args.threshold <= 1 or not 2 <= args.steps <= 1000 or
            args.per_fault < 1 or args.num_workers < 0):
        parser.error("CUDA, threshold [0,1], 2–1000 steps, and positive samples required")

    model, identity, saved = load_model(
        args.lpcr_checkpoint, args.checkpoint, args.device)
    training_loss = identity.get("training_loss")
    if (training_loss not in ("epsilon_mse_only",
                              "epsilon_mse_plus_generated_standard_bce") or
            identity.get("fault_region_policy") is None or
            identity.get("sampler", "gaussian_ddim") != "gaussian_ddim"):
        parser.error("Expected a fault-gated conditional Gaussian DDIM checkpoint")
    training_label = ("MSE-only" if training_loss == "epsilon_mse_only" else
                      f"MSE + generated BCE (weight "
                      f"{identity['decoder_bce_weight']:g})")
    full_loader, frame_ids, manifest = deployment_loader(
        args.vod_root, args.data_cache_root, args.support_cache_root,
        args.faulty_root, args.split, identity, args.num_workers,
        fault_region_root=args.fault_region_root)
    chosen = set(balanced_frame_ids(
        frame_ids, manifest["frame_faults"], args.per_fault, args.seed))
    indices = [index for index, frame_id in enumerate(frame_ids)
               if frame_id in chosen]
    loader = DataLoader(
        Subset(full_loader.dataset, indices), batch_size=identity["batch_size"],
        shuffle=False, num_workers=args.num_workers, collate_fn=_rows,
        pin_memory=True)
    alpha_bars = cosine_alpha_bars(device=args.device)
    threshold_index = round(args.threshold * THRESHOLD_STEPS)
    records = []
    for rows in loader:
        if any("clean_xyz" in row or "clean_lidar" in row for row in rows):
            raise RuntimeError("Prediction rows unexpectedly contain clean LiDAR")
        output = predict_batch(
            model, rows, alpha_bars, device=args.device, seed=args.seed,
            steps=args.steps, sampler="gaussian_ddim")
        fine, logits = output["fine"], output["logits"]
        for batch_index, row in enumerate(rows):
            candidate, candidate_logits = _frame_candidates(
                fine, logits, batch_index, row["faulty_xyz"], args.device)
            bucket = torch.floor(candidate_logits.sigmoid() *
                                 THRESHOLD_STEPS).long().clamp(0, THRESHOLD_STEPS)
            predicted = candidate[bucket >= threshold_index].cpu()
            # Clean is read only after the model has generated its prediction.
            clean_xyz = reference_clean(
                args.data_cache_root, args.split, row["frame_id"])
            clean, faulty, completed = full_frame_voxels(
                clean_xyz, row["faulty_xyz"], predicted)
            clean_keys = encode_keys(clean, GRID.shape_zyx)
            faulty_keys = encode_keys(faulty, GRID.shape_zyx)
            completed_keys = encode_keys(completed, GRID.shape_zyx)
            added = completed[~torch.isin(completed_keys, faulty_keys)]
            added_keys = encode_keys(added, GRID.shape_zyx)
            true_added = added[torch.isin(added_keys, clean_keys)]
            false_added = added[~torch.isin(added_keys, clean_keys)]
            missing = clean[~torch.isin(clean_keys, completed_keys)]
            gate_mask = fine[:, 0] == batch_index
            gate = fine[gate_mask].cpu().clone()
            gate[:, 0] = 0
            before, after = empty_accumulator(), empty_accumulator()
            add_frame(before, faulty, clean)
            add_frame(after, completed, clean)
            records.append({
                "frame_id": row["frame_id"], "split": args.split,
                "fault": manifest["frame_faults"][row["frame_id"]],
                "epoch": int(saved["epoch"]), "threshold": args.threshold,
                "oracle_clean_latent": False,
                "clean": clean, "faulty": faulty,
                "completed": completed, "added": added, "gate": gate,
                "probability": logits[gate_mask].sigmoid().cpu().numpy(),
                "overall": completion_summary(before, after),
                "clean_xyz": voxel_centers(clean),
                "faulty_xyz": voxel_centers(faulty),
                "completed_xyz": voxel_centers(completed),
                "added_xyz": voxel_centers(added),
                "true_added_xyz": voxel_centers(true_added),
                "false_added_xyz": voxel_centers(false_added),
                "missing_xyz": voxel_centers(missing),
                "gate_xyz": voxel_centers(gate),
                "radar_xyz": _roi_radar(row["radar"])})
    if len(records) != len(chosen):
        raise RuntimeError(f"Captured {len(records)} of {len(chosen)} frames")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, record in enumerate(records):
        summary = _save_record(args.output_root, record, args.seed + index * 11)
        summaries.append(summary)
        before, after = summary["overall"]["faulty"], summary["overall"]["completed"]
        print(f"{record['frame_id']} {record['fault']}: "
              f"full-ROI IoU {before['iou']:.2%} -> {after['iou']:.2%}; "
              f"{summary['added_cells']:,} new cells", flush=True)
    (args.output_root / "summary.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "epoch": int(saved["epoch"]),
        "split": args.split, "threshold": args.threshold,
        "training_loss": training_loss,
        "decoder_bce_weight": identity.get("decoder_bce_weight"),
        "sampling_steps": args.steps, "oracle_clean_latent": False,
        "prediction_inputs": ["Gaussian_noise", "faulty_LiDAR", "radar"],
        "samples": summaries}, indent=2) + "\n", encoding="utf-8")
    links = "\n".join(
        f'<li><a target="viewer" href="{html.escape(item["html"], quote=True)}">'
        f'{html.escape(item["frame_id"])} · {html.escape(item["fault"])}'
        f'</a> · <a href="{html.escape(item["png"], quote=True)}">PNG</a></li>'
        for item in summaries)
    first = html.escape(summaries[0]["html"], quote=True)
    (args.output_root / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        f"<title>{html.escape(training_label)} Stage II completion samples</title>"
        "<style>body{margin:1rem;background:#101820;color:#edf4fa;"
        "font:16px system-ui}a{color:#88ceff}iframe{width:100%;"
        "height:85vh;border:1px solid #526474}</style>"
        f"<h1>{html.escape(training_label)} Stage II · faulty LiDAR + radar inference</h1>"
        "<p>Each sample shows faulty, added, completed, and clean occupancy. "
        "Clean LiDAR is used only for display and metrics.</p>"
        f"<ul>{links}</ul><iframe name='viewer' src='{first}' "
        "title='Stage II sample'></iframe></html>", encoding="utf-8")
    print(f"3D viewer: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
