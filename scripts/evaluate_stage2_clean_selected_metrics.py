"""Score exact/tolerant occupancy and Chamfer at calibration-best IoU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity as support_identity)
from models.radar_lidar_stage2.clean_selected_metrics import (
    add_frame, empty_accumulator, summarize)
from models.radar_lidar_stage2.geometry_gate import offset_targets
from models.radar_lidar_stage2.sparse_residual import (
    LPCR_REVISION, VERSION, SparseResidualStage2, cosine_alpha_bars)
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.train_stage2_clean_denoiser import (
    MIN_RADAR_VOXELS, RADIUS_M, REVISION, _batch, _CleanRows, _rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sweep-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available() or args.num_workers < 0:
        parser.error("CUDA is required and num-workers must be nonnegative")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity = saved.get("identity")
    if not isinstance(identity, dict) or identity.get("revision") != REVISION:
        raise ValueError("Expected a clean 12-voxel/1.6 m denoiser checkpoint")
    sweep = json.loads(args.sweep_json.read_text(encoding="utf-8"))
    if (sweep.get("checkpoint") != str(args.checkpoint) or
            sweep.get("checkpoint_epoch") != saved.get("epoch") or
            sweep.get("model_revision") != REVISION or
            sweep.get("eval_timestep") != identity["eval_timestep"]):
        raise ValueError("Threshold sweep does not match this checkpoint")
    selected = sweep["selected"]["best_iou"]
    if selected is None:
        raise ValueError("No calibration-best IoU threshold in sweep")
    threshold = float(selected["threshold"])
    step_index = round(threshold * int(sweep["threshold_steps"]))
    if not 0 <= step_index <= 1000 or sweep["threshold_steps"] != 1000:
        raise ValueError("Unexpected threshold resolution")
    if identity["lpcr_sha256"] != sha256(args.lpcr_checkpoint):
        raise ValueError("Pretrained LPCR checkpoint differs from training")
    for split in ("train", "val"):
        if identity["data_manifest_sha256"][split] != sha256(
                args.data_cache_root / split / "manifest.json"):
            raise ValueError(f"Fixed {split} frames differ from training")
        if identity["support_manifest_sha256"][split] != sha256(
                args.support_cache_root / split / "manifest.json"):
            raise ValueError(f"Radar support {split} differs from training")

    saved_lpcr = torch.load(args.lpcr_checkpoint, map_location="cpu",
                            weights_only=False)
    if (saved_lpcr.get("version") != VERSION or
            saved_lpcr.get("model_revision") != LPCR_REVISION or
            saved_lpcr.get("phase") != "lpcr" or
            not saved_lpcr.get("lpcr_full_region")):
        raise ValueError("Expected the frozen full-region LPCR checkpoint")
    model = SparseResidualStage2(diffusion_target="epsilon").to(args.device)
    missing, unexpected = model.load_state_dict(saved_lpcr["model"], strict=False)
    if unexpected or any(not key.startswith("addition_head.") for key in missing):
        raise ValueError("LPCR checkpoint differs from the architecture")
    model.diffusion.load_state_dict(saved["diffusion"])
    model.eval().requires_grad_(False)

    dataset = paired_dataset(args.vod_root, "val", identity["radar_variant"],
                             include_clean=True,
                             data_cache_root=args.data_cache_root)
    frame_ids = [frame.frame_id for frame in dataset.frames]
    calibration_ids = set(sweep["calibration_frame_ids"])
    heldout_ids = set(sweep["heldout_frame_ids"])
    if (calibration_ids & heldout_ids or
            calibration_ids | heldout_ids != set(frame_ids) or
            len(calibration_ids) != sweep["frames"]["calibration"] or
            len(heldout_ids) != sweep["frames"]["heldout"]):
        raise ValueError("Threshold split differs from this validation dataset")
    expected = support_identity(
        "val", frame_ids, identity["radar_variant"],
        data_cache_root=args.data_cache_root,
        radius_m=RADIUS_M, min_radar_voxels=MIN_RADAR_VOXELS)
    cache = RadarSupportCache(args.support_cache_root / "val", expected)
    cache.check_complete()
    loader = DataLoader(
        _CleanRows(dataset, cache, None), batch_size=identity["batch_size"],
        shuffle=False, num_workers=args.num_workers, collate_fn=_rows,
        pin_memory=True, persistent_workers=args.num_workers > 0)
    alpha_bars = cosine_alpha_bars(device=args.device)
    timestep = int(identity["eval_timestep"])
    generator = torch.Generator(device=args.device)
    generator.manual_seed(int(identity["seed"]))
    accumulators = {"calibration": empty_accumulator(),
                    "heldout": empty_accumulator()}
    with torch.no_grad():
        for rows in tqdm(loader, desc="Clean denoiser selected metrics",
                         unit="batch"):
            hierarchy, clean, point_ids = _batch(rows, args.device)
            fine, coarse = hierarchy[0], hierarchy[-1]
            if len(coarse):
                x0, _ = model.lidar_encoder(clean, coarse, point_ids)
                alpha = alpha_bars[timestep]
                epsilon = torch.randn(x0.shape, dtype=x0.dtype,
                                      device=args.device, generator=generator)
                xt = alpha.sqrt() * x0 + (1 - alpha).sqrt() * epsilon
                prediction = model.diffusion(
                    xt, torch.zeros_like(x0), torch.zeros_like(x0),
                    x0.new_ones((len(x0), 1)),
                    torch.zeros(len(x0), dtype=torch.bool, device=args.device),
                    torch.full((len(rows),), timestep, device=args.device,
                               dtype=torch.long), coarse)
                recovered = ((xt - (1 - alpha).sqrt() * prediction)
                             / alpha.sqrt())
                logits, _ = model.lpcr(recovered.clamp(-1, 1), hierarchy,
                                       x0.new_ones((len(fine), 1)))
                occupied, _, _ = offset_targets(clean, fine, point_ids)
                selected_cells = (torch.floor(logits.sigmoid() * 1000)
                                  .long().clamp(0, 1000) >= step_index)
            else:
                occupied = torch.zeros(len(fine), dtype=torch.bool,
                                       device=args.device)
                selected_cells = occupied
            for frame, row in enumerate(rows):
                frame_cells = fine[:, 0] == frame
                predicted = fine[frame_cells & selected_cells].clone()
                target = fine[frame_cells & occupied].clone()
                predicted[:, 0] = 0
                target[:, 0] = 0
                name = ("calibration" if row["frame_id"] in calibration_ids
                        else "heldout")
                add_frame(accumulators[name], predicted, target)

    all_counts = {key: accumulators["calibration"][key] +
                  accumulators["heldout"][key]
                  for key in accumulators["calibration"]}
    metrics = {"calibration": summarize(accumulators["calibration"]),
               "heldout": summarize(accumulators["heldout"]),
               "all_validation": summarize(all_counts)}
    for name, row in metrics.items():
        selected_row = selected[name]
        if any(row[actual] != selected_row[expected] for actual, expected in (
                ("exact_tp", "tp"), ("predicted", "predicted"),
                ("eligible_target", "target"))):
            raise RuntimeError(f"{name} exact counts differ from threshold sweep")
    report = {
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": int(saved["epoch"]),
        "selected_by": "maximum_calibration_exact_eligible_iou",
        "occupancy_threshold": threshold,
        "sampling": "one_step_clean_x0_at_fixed_t",
        "eval_timestep": timestep,
        "target": "clean_LiDAR_occupied_voxel_centers_inside_12vox_1p6m_gate",
        "prediction": "denoised_LPCR_occupied_voxel_centers_inside_12vox_1p6m_gate",
        "tolerant_matching": "one_to_one_voxel_center_distance_at_most_0p2m",
        "chamfer_definition": ("per-frame mean nearest-center L2 in both "
                               "directions, summed; paired nonempty frames only"),
        "validation_note": sweep["validation_note"],
        "metrics": metrics}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(args.output_json)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Checkpoint epoch {saved['epoch']}; best exact-IoU threshold "
          f"{threshold:.3f}", flush=True)
    for name, row in metrics.items():
        chamfer = row["chamfer_squared_l2_m2_paired_frame_mean"]
        chamfer_text = "undefined" if chamfer is None else f"{chamfer:.6f} m²"
        print(f"{name} ({row['frames']} frames): IoU={row['iou']:.2%}, "
              f"IoU@0.2m={row['iou_0p2m']:.2%}, F1={row['f1']:.2%}, "
              f"F1@0.2m={row['f1_0p2m']:.2%}, "
              f"Chamfer squared L2={chamfer_text}, "
              f"Chamfer paired={row['chamfer_paired_frames']}, "
              f"empty predictions={row['empty_prediction_frames']}", flush=True)
    print(f"Report: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
