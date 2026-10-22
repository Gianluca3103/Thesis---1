"""Verify the three severe Stage-II fault inputs on one VoD frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.fault_data import (
    FAULT_NAMES, SevereFogCache, selected_fault, severe_fault,
)
from models.radar_lidar_stage2.sparse_geometry import GRID
from scripts.predict_stage2_sparse import _present_in_faulty
from scripts.train_stage2_sparse import _combine_points


def _roi_count(points: torch.Tensor) -> int:
    xyz = points[:, :3]
    minimum = xyz.new_tensor(GRID.minimum_xyz)
    maximum = xyz.new_tensor(GRID.maximum_xyz)
    return int(((xyz >= minimum) & (xyz < maximum)).all(1).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--fault-cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3103)
    args = parser.parse_args()
    dataset = paired_dataset(args.vod_root, args.split, "radar_5frames",
                             include_clean=True,
                             data_cache_root=args.data_cache_root)
    if not 0 <= args.frame_index < len(dataset):
        parser.error("Frame index is outside the selected split")
    cache = SevereFogCache(args.fault_cache_root, args.split,
                           [frame.frame_id for frame in dataset.frames],
                           seed=args.seed, check_complete=False)
    row = dataset[args.frame_index]
    clean = row["clean_lidar"]
    counts = {"frame_id": row["frame_id"], "clean_points": len(clean),
              "clean_stage2_roi": _roi_count(clean), "faults": {}}
    faulty_clouds = []
    for name in FAULT_NAMES:
        faulty, actual = severe_fault(clean, row["frame_id"], 0, cache,
                                      seed=args.seed, fault_type=name)
        if actual != name or faulty.ndim != 2 or faulty.shape[1] != 4:
            raise AssertionError(f"Invalid {name} output")
        if not torch.isfinite(faulty).all():
            raise AssertionError(f"Nonfinite {name} output")
        counts["faults"][name] = {"points": len(faulty),
                                  "stage2_roi": _roi_count(faulty)}
        faulty_clouds.append(faulty)
    if counts["faults"]["total_loss"]["points"] != 0:
        raise AssertionError("Total-loss injector retained LiDAR returns")
    if counts["faults"]["fov_filter"]["points"] >= len(clean):
        raise AssertionError("Severity-5 FOV injector did not narrow this frame")
    fog = faulty_clouds[FAULT_NAMES.index("fog_sim")]
    if len(fog) == len(clean) and torch.equal(fog, clean):
        raise AssertionError("Severity-5 fog injector did not change the scan")
    fog_xyz_kept = _present_in_faulty(clean[:, :3].numpy(), fog[:, :3].numpy())
    fog_full_kept = _present_in_faulty(clean.numpy(), fog.numpy())
    counts["faults"]["fog_sim"]["clean_xyz_absent_or_shifted"] = int(
        (~fog_xyz_kept).sum())
    counts["faults"]["fog_sim"]["clean_reflectivity_changed"] = int(
        (fog_xyz_kept & ~fog_full_kept).sum())
    if not counts["faults"]["fog_sim"]["clean_reflectivity_changed"]:
        raise AssertionError("Severity-5 fog did not alter reflectivity")
    combined, batch_ids = _combine_points(faulty_clouds)
    if len(combined) != sum(len(cloud) for cloud in faulty_clouds):
        raise AssertionError("Empty total-loss frame broke physical batching")
    if batch_ids is None or int((batch_ids == 0).sum()) != 0:
        raise AssertionError("Empty total-loss frame received LiDAR points")
    observed = {selected_fault(row["frame_id"], epoch, args.seed)
                for epoch in range(3)}
    if observed != set(FAULT_NAMES):
        raise AssertionError("Three training epochs do not cover every severe fault")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
