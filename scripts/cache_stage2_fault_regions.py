"""Cache the mild injector's untouched/touched regions without resimulating fog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.fault_region import (
    FaultRegionCache, cache_identity, partial_azimuth_region)
from models.radar_lidar_stage2.mild_fault_data import (
    MILD_FOG2_VERSION, frame_seed, partial_azimuth_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant",
                        default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--split", choices=("train", "val", "test", "both", "all"), default="both")
    args = parser.parse_args()
    splits = (("train", "val", "test") if args.split == "all" else
              ("train", "val") if args.split == "both" else (args.split,))
    for split in splits:
        dataset = paired_dataset(args.vod_root, split, args.radar_variant,
                                 include_clean=True,
                                 data_cache_root=args.data_cache_root)
        ids = [frame.frame_id for frame in dataset.frames]
        mild_path = args.faulty_root / split / "manifest.json"
        mild = json.loads(mild_path.read_text(encoding="utf-8"))
        if (mild.get("version") != MILD_FOG2_VERSION
                or mild.get("radar_variant") != args.radar_variant
                or mild.get("frame_ids") != ids):
            raise ValueError(f"Mild-fault manifest differs from {split} fixed frames")
        expected = cache_identity(
            split, ids, mild_path,
            args.data_cache_root / split / "manifest.json")
        cache = FaultRegionCache(args.output_root / split, expected, create=True)
        if cache.regions is not None:
            print(f"{split}: reused {len(cache.regions)} injector footprints", flush=True)
            continue
        regions = {}
        counts = {fault: 0 for fault in mild["fault_types"]}
        for frame_id in tqdm(ids, desc=f"Injector footprints {split}", unit="frame"):
            if not (args.faulty_root / split / f"{frame_id}.npz").is_file():
                raise FileNotFoundError(f"Missing cached mild-fault frame: {split}/{frame_id}")
            fault = mild["frame_faults"][frame_id]
            counts[fault] += 1
            if fault == "fog_sim_2":
                # Fog was applied to the full raw scan before cropping.
                region = {"kind": "full_scan"}
            elif fault in ("partial_loss_30", "partial_loss_40", "partial_loss_50"):
                saved = torch.load(dataset.path(frame_id, "clean_lidar"),
                                   map_location="cpu", weights_only=True)
                if saved.get("frame_id") != frame_id:
                    raise ValueError(f"Fixed clean frame mismatch: {frame_id}")
                clean = saved["points"][:, :3].numpy()
                fraction = int(fault.rsplit("_", 1)[1]) / 100
                seed = frame_seed(int(mild["seed"]), split, frame_id)
                reconstructed_faulty = partial_azimuth_loss(clean, fraction, seed)
                cached_faulty = load_faulty(args.faulty_root, split, frame_id).numpy()
                if not np.array_equal(reconstructed_faulty, cached_faulty):
                    raise ValueError(f"Cached partial fault differs from injector: {frame_id}")
                region = partial_azimuth_region(clean, fraction, seed)
            else:
                raise ValueError(f"Unsupported mild fault: {fault}")
            regions[frame_id] = {"fault_type": fault, **region}
        cache.write(regions)
        print(f"{split}: cached {len(regions)} injector footprints, {counts}",
              flush=True)


if __name__ == "__main__":
    main()
