"""Build the same version-3 geometry cache using parallel CPU geometry workers.

Sparse encoder and unpruned decoder coordinates depend only on input voxel
coordinates. Their learned feature convolutions are unnecessary for cache
construction. Verify against entries from the original builder before use.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage1.correspondence import Neighborhood
from models.radar_lidar_stage1.sparse import (SparseSites, decode_keys,
                                               encode_keys, voxelize)
from models.radar_lidar_stage2.candidate_domain import CandidateDomain
from models.radar_lidar_stage2.voxel_target import make_targets
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.meta_cache import (MetaOccupancyCache,
                                                        cache_manifest)
from models.radar_lidar_crossmodal_v1.meta_occupancy import (
    MetaOccupancyConfig, candidate_support, supported_clean_points,
)
from models.radar_lidar_crossmodal_v1.pipeline_data import collate_stage1, paired_dataset


COARSE_SHAPE = GRID.scale_shape(8)
AXIAL = torch.tensor(((0, 0, 0), (1, 0, 0), (-1, 0, 0),
                      (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)))
CHILD_OFFSETS = torch.cartesian_prod(*(torch.arange(8) for _ in range(3)))


def bottleneck_geometry(points: torch.Tensor, valid: torch.Tensor) -> SparseSites:
    """Exact support of three stride-2 downsamples and final axial growth."""
    fine, _, _, _ = voxelize(points, valid, GRID)
    parent = fine.clone()
    parent[:, 1:] //= 8
    base = decode_keys(torch.unique(encode_keys(parent, COARSE_SHAPE), sorted=True),
                       COARSE_SHAPE)
    if len(base):
        expanded = base[:, None, :].expand(-1, len(AXIAL), -1).clone()
        expanded[:, :, 1:] += AXIAL[None, :, :].to(expanded.device)
        expanded = expanded.reshape(-1, 4)
        limits = expanded.new_tensor(COARSE_SHAPE)
        inside = ((expanded[:, 1:] >= 0) & (expanded[:, 1:] < limits)).all(-1)
        keys = torch.unique(encode_keys(expanded[inside], COARSE_SHAPE), sorted=True)
        coarse = decode_keys(keys, COARSE_SHAPE)
    else:
        coarse = base
    features = points.new_zeros((len(coarse), 1))
    return SparseSites._from_validated(coarse, features, COARSE_SHAPE, 8)


def candidate_geometry(radar: SparseSites) -> torch.Tensor:
    """All 8^3 children per stride-8 site, sorted like the raw decoder."""
    if not len(radar.coords):
        return radar.coords.new_empty((0, 4))
    children = radar.coords[:, None, :].expand(-1, len(CHILD_OFFSETS), -1).clone()
    children[:, :, 1:] = (radar.coords[:, None, 1:] * 8
                          + CHILD_OFFSETS[None, :, :].to(children.device))
    keys = encode_keys(children.reshape(-1, 4), GRID.shape_zyx)
    return decode_keys(torch.sort(keys).values, GRID.shape_zyx)


def _cpu_neighbors(radar: SparseSites, lidar: SparseSites,
                   radius_m: float, max_candidates: int) -> Neighborhood:
    """The original KD-tree query with one thread per parallel cache worker."""
    n = len(radar.coords)
    indices = torch.full((n, max_candidates), -1, dtype=torch.long)
    distances = torch.full((n, max_candidates), float("inf"), dtype=torch.float32)
    if not n or not len(lidar.coords):
        return Neighborhood(indices, distances)
    radar_xyz = radar.centers_xyz(GRID).numpy()
    lidar_xyz = lidar.centers_xyz(GRID).numpy()
    radar_batch = radar.coords[:, 0].numpy()
    lidar_batch = lidar.coords[:, 0].numpy()
    for batch in np.unique(radar_batch):
        query = np.flatnonzero(radar_batch == batch)
        reference = np.flatnonzero(lidar_batch == batch)
        if not len(reference):
            continue
        delta, local = cKDTree(lidar_xyz[reference]).query(
            radar_xyz[query], k=max_candidates,
            distance_upper_bound=radius_m, workers=1)
        delta = np.asarray(delta).reshape(len(query), max_candidates)
        local = np.asarray(local).reshape(len(query), max_candidates)
        present = np.isfinite(delta)
        mapped = np.full(local.shape, -1, dtype=np.int64)
        mapped[present] = reference[local[present]]
        indices[torch.as_tensor(query)] = torch.as_tensor(mapped)
        distances[torch.as_tensor(query)] = torch.as_tensor(delta, dtype=torch.float32)
    return Neighborhood(indices, distances)


@torch.no_grad()
def build_entry_fast(batch: dict, config: MetaOccupancyConfig) -> dict:
    radar = bottleneck_geometry(batch["radar"], batch["radar_valid"])
    lidar = bottleneck_geometry(batch["clean_lidar"], batch["clean_lidar_valid"])
    candidate = candidate_geometry(radar)
    domain = CandidateDomain(candidate, torch.zeros(len(candidate)), GRID, {})
    targets = make_targets(domain, batch["clean_lidar"], batch["clean_lidar_valid"])
    support = candidate_support(candidate, batch["radar"], batch["radar_valid"],
                                config.radius_m)
    observed, _, _, _ = voxelize(batch["clean_lidar"], batch["clean_lidar_valid"], GRID)
    observed = observed[candidate_support(observed, batch["radar"],
                                          batch["radar_valid"], config.radius_m)]
    neighbors = _cpu_neighbors(radar, lidar, config.radius_m, config.max_candidates)
    clean_supported = supported_clean_points(
        batch["clean_lidar"], batch["clean_lidar_valid"],
        batch["radar"], batch["radar_valid"], config.radius_m)
    return {
        "radar_coords": radar.coords.to(torch.int32),
        "lidar_coords": lidar.coords.to(torch.int32),
        "neighbor_indices": neighbors.indices.to(torch.int32),
        "neighbor_distances": neighbors.distances_m,
        "candidate_coords": candidate.to(torch.int32),
        "occupied": targets.occupied,
        "known_free": targets.known_free,
        "support": support,
        "observed_coords": observed.to(torch.int32),
        "clean_supported_points": clean_supported,
        "represented_clean_points_supported": int(targets.clean_point_count[support].sum()),
    }


def _build_collate(rows: list[dict], config: MetaOccupancyConfig):
    os.environ["STAGE2_KDTREE_WORKERS"] = "1"
    batch = collate_stage1(rows)
    return batch["frame_id"][0], build_entry_fast(batch, config)


def _verify_existing(dataset, cache, count: int, config: MetaOccupancyConfig):
    for index in tqdm(range(count), desc="Verify fast geometry", unit="frame"):
        frame_id = dataset.frames[index].frame_id
        if not cache.path(frame_id).is_file():
            raise FileNotFoundError(f"Existing reference cache frame missing: {frame_id}")
        old = cache[index]
        os.environ["STAGE2_KDTREE_WORKERS"] = "1"
        fresh = build_entry_fast(collate_stage1([dataset[index]]), config)
        for name, value in fresh.items():
            original = old[name]
            if isinstance(value, torch.Tensor):
                if not torch.equal(value, original):
                    raise ValueError(f"Fast geometry differs for {frame_id} {name}")
            elif value != original:
                raise ValueError(f"Fast geometry differs for {frame_id} {name}")
    print(f"Verified all fields for {count} existing frames", flush=True)


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
    parser.add_argument("--verify-existing", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1 or args.verify_existing < 0
            or args.num_workers < 1):
        parser.error("Limit and worker count must be positive; verification count nonnegative")
    config = MetaOccupancyConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    dataset = paired_dataset(args.vod_root, args.split, args.radar_variant,
                             data_cache_root=args.data_cache_root)
    ids = [frame.frame_id for frame in dataset.frames]
    manifest = cache_manifest(
        split=args.split, frame_ids=ids, vod_root=args.vod_root,
        radar_variant=args.radar_variant, radar_encoder=args.radar_encoder,
        lidar_encoder=args.lidar_encoder, radius_m=config.radius_m,
        max_candidates=config.max_candidates)
    cache = MetaOccupancyCache(args.cache_root / args.split, manifest, create=True)
    if args.verify_existing:
        _verify_existing(dataset, cache, min(args.verify_existing, len(dataset)), config)
    selected = ids[:args.limit] if args.limit is not None else ids
    missing = [index for index, frame_id in enumerate(selected)
               if not cache.path(frame_id).is_file()]
    if not missing:
        print(f"{args.split}: all {len(selected)} selected frames are cached", flush=True)
        return
    loader = DataLoader(Subset(dataset, missing), batch_size=1,
                        num_workers=args.num_workers,
                        collate_fn=partial(_build_collate, config=config),
                        pin_memory=False, prefetch_factor=2)
    written = 0
    for count, (frame_id, entry) in enumerate(
            tqdm(loader, desc=f"Fast cache {args.split}", unit="frame"), 1):
        written += cache.write(frame_id, entry)
        if count == min(16, len(missing)):
            remaining = written / count * (len(missing) - count)
            free = shutil.disk_usage(cache.root).free
            print(f"Compressed cache estimate: {remaining / 2**30:.2f} GiB remaining; "
                  f"{free / 2**30:.1f} GiB free", flush=True)
            if remaining * 1.2 > free:
                raise OSError("Insufficient space for estimated compressed cache")
    if args.limit is None:
        cache.check_complete()
    print(f"{args.split}: wrote {len(missing)} frames; "
          f"{written / 2**30:.3f} GiB new cache data", flush=True)


if __name__ == "__main__":
    main()
