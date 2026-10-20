"""Precompute fixed geometry and labels; training recomputes encoder features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.radar_lidar_stage1.correspondence import local_neighbors
from models.radar_lidar_stage1.sparse import voxelize
from models.radar_lidar_stage2.candidate_domain import CandidateDomain
from models.radar_lidar_stage2.voxel_target import make_targets
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.meta_cache import MetaOccupancyCache, cache_manifest
from models.radar_lidar_crossmodal_v1.meta_occupancy import (
    MetaOccupancyConfig, candidate_support, supported_clean_points,
)
from models.radar_lidar_crossmodal_v1.pipeline_data import collate_stage1, paired_dataset
from models.radar_lidar_crossmodal_v1.previous_encoder import FrozenPreviousStage1Sparse
from models.radar_lidar_crossmodal_v1.student import RadarOnlyStudent, StudentConfig
from models.radar_lidar_crossmodal_v1.teacher import tiled_local_neighbors


@torch.no_grad()
def build_entry(student, lidar_encoder, raw, config, device):
    batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor)
             else value for key, value in raw.items()}
    radar = student.radar_encoder(batch["radar"], batch["radar_valid"])
    lidar = lidar_encoder(batch["clean_lidar"], batch["clean_lidar_valid"])
    output = student.forward_encoded(radar)
    candidate = output.confidence_logits.coords
    domain = CandidateDomain(candidate, torch.zeros(len(candidate), device=device), GRID, {})
    targets = make_targets(domain, batch["clean_lidar"], batch["clean_lidar_valid"])
    support = candidate_support(candidate, batch["radar"], batch["radar_valid"],
                                config.radius_m)
    observed, _, _, _ = voxelize(batch["clean_lidar"], batch["clean_lidar_valid"], GRID)
    observed = observed[candidate_support(observed, batch["radar"],
                                          batch["radar_valid"], config.radius_m)]
    if radar.coords.is_cuda and os.environ.get("CROSSMODAL_GPU_NEIGHBORS") == "1":
        neighbors = tiled_local_neighbors(radar, lidar, config.radius_m,
                                          config.max_candidates)
    else:
        neighbors = local_neighbors(radar, lidar, GRID, config.radius_m,
                                    config.max_candidates)
    clean_supported = supported_clean_points(
        batch["clean_lidar"], batch["clean_lidar_valid"],
        batch["radar"], batch["radar_valid"], config.radius_m)
    return {
        "radar_coords": radar.coords.cpu().to(torch.int32),
        "lidar_coords": lidar.coords.cpu().to(torch.int32),
        "neighbor_indices": neighbors.indices.cpu().to(torch.int32),
        "neighbor_distances": neighbors.distances_m.cpu(),
        "candidate_coords": candidate.cpu().to(torch.int32),
        "occupied": targets.occupied.cpu(),
        "known_free": targets.known_free.cpu(),
        "support": support.cpu(),
        "observed_coords": observed.cpu().to(torch.int32),
        "clean_supported_points": clean_supported,
        "represented_clean_points_supported": int(targets.clean_point_count[support].sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--radar-encoder", type=Path, required=True)
    parser.add_argument("--lidar-encoder", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/crossmodal_meta_occupancy.json"))
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    config = MetaOccupancyConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    dataset = paired_dataset(args.vod_root, args.split, args.radar_variant,
                             data_cache_root=args.data_cache_root)
    frame_ids = [frame.frame_id for frame in dataset.frames]
    manifest = cache_manifest(
        split=args.split, frame_ids=frame_ids, vod_root=args.vod_root,
        radar_variant=args.radar_variant, radar_encoder=args.radar_encoder,
        lidar_encoder=args.lidar_encoder, radius_m=config.radius_m,
        max_candidates=config.max_candidates)
    cache = MetaOccupancyCache(args.cache_root / args.split, manifest, create=True)
    selected = frame_ids[:args.limit] if args.limit is not None else frame_ids
    missing_indices = [index for index, frame_id in enumerate(selected)
                       if not cache.path(frame_id).is_file()]
    if not missing_indices:
        print(f"{args.split}: all {len(selected)} selected frames are cached", flush=True)
        return
    from torch.utils.data import Subset
    loader = DataLoader(Subset(dataset, missing_indices), batch_size=1,
                        num_workers=args.num_workers, collate_fn=collate_stage1,
                        pin_memory=True)
    student = RadarOnlyStudent(str(args.radar_encoder),
                               StudentConfig(positive_weight=1, dice_weight=0,
                                             lambda_teacher_evidence=0,
                                             decoder_from_conditioned=True)).to(args.device).eval()
    lidar_encoder = FrozenPreviousStage1Sparse(args.lidar_encoder,
                                                modality="lidar").to(args.device).eval()
    new_bytes = 0
    for count, raw in enumerate(tqdm(loader, desc=f"Meta cache {args.split}", unit="frame"), 1):
        frame_id = raw["frame_id"][0]
        entry = build_entry(student, lidar_encoder, raw, config, args.device)
        new_bytes += cache.write(frame_id, entry)
        if count == min(16, len(missing_indices)):
            estimated = new_bytes / count * (len(missing_indices) - count)
            free = shutil.disk_usage(cache.root).free
            print(f"Cache estimate: {estimated / 2**30:.1f} GiB remaining; "
                  f"{free / 2**30:.1f} GiB free", flush=True)
            if estimated * 1.2 > free:
                raise OSError("Insufficient space for estimated meta cache; partial entries can be reused")
    if args.limit is None:
        cache.check_complete()
    print(f"{args.split}: wrote {len(missing_indices)} frames; "
          f"{new_bytes / 2**30:.2f} GiB new cache data", flush=True)


if __name__ == "__main__":
    main()
