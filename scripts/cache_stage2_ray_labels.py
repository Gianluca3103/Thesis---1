"""Cache fixed Stage-II ray associations before diffusion training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity as support_identity)
from models.radar_lidar_stage2.addition import addition_context
from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.fault_region import (
    FaultRegionCache, cache_identity as region_identity)
from models.radar_lidar_stage2.geometry_gate import (
    RADIUS_M, geometry_gate_output, offset_targets)
from models.radar_lidar_stage2.ray_label_cache import (
    RayLabelCache, identity as ray_identity)
from models.radar_lidar_stage2.ray_loss import addition_ray_losses
from models.radar_lidar_stage2.sparse_geometry import select_allowed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--fault-region-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant",
                        default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--split", choices=("train", "val", "both"),
                        default="both")
    parser.add_argument("--gate-threshold", type=float, default=0.25)
    parser.add_argument("--ray-angular-tolerance-deg", type=float, default=1.5)
    parser.add_argument("--ray-lateral-tolerance-m", type=float, default=0.15)
    parser.add_argument("--ray-range-margin-m", type=float)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="Use CUDA to exactly match training gate coordinates")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required for the training-matched ray cache")
    if (not 0 < args.gate_threshold < 1
            or args.ray_angular_tolerance_deg <= 0
            or args.ray_lateral_tolerance_m <= 0
            or args.ray_range_margin_m is not None
            and args.ray_range_margin_m < 0):
        parser.error("Invalid gate or ray-matching tolerance")
    splits = ("train", "val") if args.split == "both" else (args.split,)
    for split in splits:
        dataset = paired_dataset(args.vod_root, split, args.radar_variant,
                                 include_clean=True,
                                 data_cache_root=args.data_cache_root)
        ids = [frame.frame_id for frame in dataset.frames]
        mild_manifest = json.loads((args.faulty_root / split / "manifest.json")
                                   .read_text(encoding="utf-8"))
        if (mild_manifest.get("frame_ids") != ids
                or mild_manifest.get("radar_variant") != args.radar_variant):
            raise ValueError(f"Mild-fault cache differs from {split} fixed frames")
        support = RadarSupportCache(
            args.support_cache_root / split,
            support_identity(split, ids, args.radar_variant,
                             data_cache_root=args.data_cache_root,
                             radius_m=RADIUS_M))
        support.check_complete()
        regions = FaultRegionCache(
            args.fault_region_root / split,
            region_identity(split, ids,
                            args.faulty_root / split / "manifest.json",
                            args.data_cache_root / split / "manifest.json"))
        if regions.regions is None:
            raise FileNotFoundError(f"Missing {split} injector footprints")
        cache = RayLabelCache(
            args.output_root / split,
            ray_identity(
                split, ids, data_cache_root=args.data_cache_root,
                faulty_root=args.faulty_root,
                support_cache_root=args.support_cache_root,
                fault_region_root=args.fault_region_root,
                radar_variant=args.radar_variant,
                gate_threshold=args.gate_threshold,
                angular_tolerance_deg=args.ray_angular_tolerance_deg,
                lateral_tolerance_m=args.ray_lateral_tolerance_m,
                range_margin_m=args.ray_range_margin_m,
                device=args.device), create=True)
        reused = written = hit_count = extra_count = 0
        for index, frame_id in enumerate(tqdm(ids, desc=f"Ray labels {split}",
                                              unit="frame")):
            if cache.path(frame_id).is_file():
                reused += 1
                continue
            row = dataset[index]
            if row["frame_id"] != frame_id:
                raise ValueError(f"Fixed-frame order differs: {frame_id}")
            radar = row["radar"].to(args.device)
            clean = row["clean_lidar"][:, :3].to(args.device)
            faulty = load_faulty(args.faulty_root, split, frame_id).to(args.device)
            with torch.no_grad():
                output = geometry_gate_output(
                    radar, faulty, regions.get(frame_id), support.load(frame_id))
                fine, _ = select_allowed(
                    output["confidence_coords"],
                    output["confidence_probabilities"], args.gate_threshold)
                if len(fine):
                    occupied, _, _ = offset_targets(clean, fine)
                    observed = addition_context(faulty, fine).observed
                    result = addition_ray_losses(
                        fine, torch.zeros(len(fine), device=args.device),
                        observed, occupied, clean,
                        angular_tolerance_deg=args.ray_angular_tolerance_deg,
                        lateral_tolerance_m=args.ray_lateral_tolerance_m,
                        range_margin_m=args.ray_range_margin_m,
                        collect_labels=True)
                    labels = result["labels"][0]
                    hit_count += result["hit_rays"]
                    extra_count += result["extra_sites"]
                else:
                    empty = torch.empty(0, dtype=torch.int32).numpy()
                    labels = {"candidate_count": 0, "hit": empty,
                              "front": empty, "extra": empty}
                cache.write(frame_id, fine, labels)
                written += 1
        cache.check_complete()
        print(f"{split}: {written} ray files written, {reused} reused; "
              f"new hit/extra labels={hit_count:,}/{extra_count:,}", flush=True)


if __name__ == "__main__":
    main()
