"""Sample every validation frame and score additions inside its eligible gate."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from tqdm import tqdm

from models.radar_lidar_stage1.sparse import encode_keys

from .eligible_occupancy_metrics import occupancy_counts, summarize_counts
from .fault_data import load_faulty
from .geometry_gate import geometry_gate_output, offset_targets
from .sparse_geometry import GRID


def _empty_counts() -> dict[str, int]:
    return {"predicted": 0, "eligible_target": 0,
            "exact_tp": 0, "tolerant_tp_0p2m": 0}


@torch.no_grad()
def evaluate_all_eligible(model, dataset, *, faulty_root: Path, support_cache,
                          region_cache, fault_names: dict[str, str],
                          alpha_bars: torch.Tensor, gate_threshold: float,
                          occupancy_threshold: float, sampling_steps: int,
                          device: str, min_radar_voxels: int = 3) -> dict:
    """Micro IoU over all gated missing-clean voxels, using sampled occupancy."""
    if (not 0 < occupancy_threshold < 1 or sampling_steps < 1
            or sampling_steps > len(alpha_bars)):
        raise ValueError("Invalid eligible validation threshold or diffusion steps")
    if len(dataset) != len(dataset.frames):
        raise ValueError("Full eligible validation needs the untruncated dataset")
    model.eval()
    totals = _empty_counts()
    by_fault = {name: _empty_counts() for name in sorted(set(fault_names.values()))}
    candidate_cells = 0
    # Sampling uses a fixed seed per frame while restoring the training RNG.
    rng_devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        for index in tqdm(range(len(dataset)), desc="Full eligible Stage-II validation",
                          unit="frame", dynamic_ncols=True):
            row = dataset[index]
            frame_id = row["frame_id"]
            radar = row["radar"].to(device)
            faulty = load_faulty(faulty_root, "val", frame_id).to(device)
            clean = row["clean_lidar"][:, :3].to(device)
            output = geometry_gate_output(
                radar, faulty, region_cache.get(frame_id),
                support_cache.load(frame_id), min_radar_voxels)
            conditions = model.conditions(output, gate_threshold)
            fine = conditions.hierarchy[0]
            candidate_cells += len(fine)
            if len(fine):
                occupied, _, _ = offset_targets(clean, fine)
                target = fine[occupied]
                seed = int.from_bytes(hashlib.sha256(
                    f"3103:{frame_id}".encode()).digest()[:8], "little") % (2**63)
                torch.manual_seed(seed)
                predicted = model.sample(
                    faulty, conditions, alpha_bars, steps=sampling_steps,
                    occupancy_threshold=occupancy_threshold,
                    missing_only=True)["occupied_coordinates"]
                if len(predicted) and not bool(torch.isin(
                        encode_keys(predicted, GRID.shape_zyx),
                        encode_keys(fine, GRID.shape_zyx)).all()):
                    raise RuntimeError(f"Prediction outside eligible cells: {frame_id}")
            else:
                target = fine
                predicted = fine
            counts = occupancy_counts(predicted, target)
            for key, value in counts.items():
                totals[key] += value
                by_fault[fault_names[frame_id]][key] += value
    return {"frames": len(dataset), "sampling_steps": sampling_steps,
            "occupancy_threshold": occupancy_threshold,
            "tolerance_m": 0.2,
            "target_definition": "clean_missing_occupied_voxels_inside_eligible_gate",
            "prediction_definition": "sampled_added_occupied_voxels_inside_eligible_gate",
            "mean_candidate_cells": candidate_cells / max(len(dataset), 1),
            **summarize_counts(totals),
            "by_fault": {name: summarize_counts(row)
                         for name, row in by_fault.items()}}
