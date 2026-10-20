"""Fixed inputs for the meta-occupancy run; no trainable feature is cached."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import torch

from models.radar_lidar_stage1.correspondence import Neighborhood
from models.radar_lidar_stage1.sparse import SparseSites

from .encoder import GRID


VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hash() -> str:
    repo = Path(__file__).resolve().parents[2]
    files = (
        "Fault_Localization_Model/vod_dataset/vod_io.py",
        "models/radar_lidar_stage1/data.py",
        "models/radar_lidar_stage1/radar_fov.py",
        "models/radar_lidar_stage1/sparse.py",
        "models/radar_lidar_stage1/correspondence.py",
        "models/radar_lidar_stage2/voxel_target.py",
        "models/radar_lidar_stage2/candidate_domain.py",
        "models/radar_lidar_crossmodal_v1/previous_encoder.py",
        "models/radar_lidar_crossmodal_v1/fixed_frame_cache.py",
        "models/radar_lidar_crossmodal_v1/student.py",
        "models/radar_lidar_crossmodal_v1/teacher.py",
        "models/radar_lidar_crossmodal_v1/meta_occupancy.py",
        "models/radar_lidar_crossmodal_v1/meta_cache.py",
        "scripts/cache_meta_occupancy.py",
    )
    digest = hashlib.sha256()
    for name in files:
        digest.update((repo / name).read_bytes())
    return digest.hexdigest()


def cache_manifest(*, split: str, frame_ids: list[str], vod_root: Path,
                   radar_variant: str, radar_encoder: Path, lidar_encoder: Path,
                   radius_m: float, max_candidates: int) -> dict:
    if split not in ("train", "val") or radius_m <= 0 or max_candidates < 1:
        raise ValueError("Invalid meta cache split or neighborhood")
    return {
        "version": VERSION, "split": split, "frame_ids": frame_ids,
        "vod_root": str(vod_root.resolve()), "radar_variant": radar_variant,
        "radar_sha256": _sha256(radar_encoder),
        "lidar_sha256": _sha256(lidar_encoder),
        "radius_m": radius_m, "max_candidates": max_candidates,
        "source_hash": _source_hash(),
        "grid_minimum_xyz": list(GRID.minimum_xyz),
        "grid_size_xyz": list(GRID.size_xyz),
        "grid_shape_zyx": list(GRID.shape_zyx),
    }


class MetaOccupancyCache:
    """Resumable, versioned cache of deterministic per-frame geometry."""

    def __init__(self, root: Path, expected: dict, *, create: bool = False):
        self.root = Path(root)
        self.expected = expected
        self.frames = expected["frame_ids"]
        manifest = self.root / "manifest.json"
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            if not manifest.exists():
                temporary = manifest.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(expected, indent=2), encoding="utf-8")
                temporary.replace(manifest)
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Meta cache manifest is missing or incompatible: {manifest}")

    def path(self, frame_id: str) -> Path:
        if frame_id not in self.frames or not frame_id.isdigit():
            raise ValueError("Frame is not in this cache split")
        return self.root / f"{frame_id}.pt"

    def check_complete(self) -> None:
        missing = [frame_id for frame_id in self.frames if not self.path(frame_id).is_file()]
        if missing:
            raise FileNotFoundError(f"Meta cache lacks {len(missing)} frames; first: {missing[0]}")

    def write(self, frame_id: str, entry: dict) -> int:
        path = self.path(frame_id)
        if path.is_file():
            self._read(frame_id)
            return path.stat().st_size
        entry = {"version": VERSION, "frame_id": frame_id, **entry}
        self._check(entry, frame_id)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            torch.save(entry, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path.stat().st_size

    def _check(self, entry: dict, frame_id: str) -> None:
        if entry.get("version") != VERSION or entry.get("frame_id") != frame_id:
            raise ValueError(f"Invalid meta cache identity: {frame_id}")
        radar, lidar, candidate = (entry[f"{name}_coords"]
                                   for name in ("radar", "lidar", "candidate"))
        observed = entry["observed_coords"]
        if (any(value.ndim != 2 or value.shape[1] != 4
                for value in (radar, lidar, candidate, observed))
                or any(bool(value[:, 0].ne(0).any())
                       for value in (radar, lidar, candidate, observed))
                or entry["radar_features"].shape != (len(radar), 64)
                or entry["lidar_features"].shape != (len(lidar), 64)
                or entry["neighbor_indices"].shape != (len(radar), self.expected["max_candidates"])
                or entry["neighbor_distances"].shape != entry["neighbor_indices"].shape
                or bool(((entry["neighbor_indices"] < -1)
                         | (entry["neighbor_indices"] >= len(lidar))).any())
                or any(entry[name].shape != (len(candidate),)
                       for name in ("occupied", "known_free", "support"))
                or any(entry[name].dtype != torch.bool
                       for name in ("occupied", "known_free", "support"))
                or bool((entry["occupied"] & entry["known_free"]).any())):
            raise ValueError(f"Invalid meta cache shapes: {frame_id}")

    def _read(self, frame_id: str) -> dict:
        entry = torch.load(self.path(frame_id), map_location="cpu", weights_only=True)
        self._check(entry, frame_id)
        return entry

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self._read(self.frames[index])

    def set_epoch(self, epoch):
        pass


@dataclass(frozen=True)
class CachedOccupancyTargets:
    occupied: torch.Tensor
    known_free: torch.Tensor
    represented_clean_points_supported: int


def collate_meta_cache(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Empty meta cache batch")

    def coordinates(name):
        return torch.cat([
            torch.cat((torch.full_like(row[name][:, :1], batch), row[name][:, 1:]), dim=1)
            for batch, row in enumerate(rows)
        ])

    offsets, offset = [], 0
    for row in rows:
        local = row["neighbor_indices"].long()
        offsets.append(torch.where(local >= 0, local + offset, local))
        offset += len(row["lidar_coords"])
    result = {
        "frame_id": [row["frame_id"] for row in rows],
        "radar_coords": coordinates("radar_coords"),
        "radar_features": torch.cat([row["radar_features"] for row in rows]),
        "lidar_coords": coordinates("lidar_coords"),
        "lidar_features": torch.cat([row["lidar_features"] for row in rows]),
        "neighbor_indices": torch.cat(offsets),
        "neighbor_distances": torch.cat([row["neighbor_distances"] for row in rows]),
        "candidate_coords": coordinates("candidate_coords"),
        "observed_coords": coordinates("observed_coords"),
        "occupied": torch.cat([row["occupied"] for row in rows]),
        "known_free": torch.cat([row["known_free"] for row in rows]),
        "support": torch.cat([row["support"] for row in rows]),
        "clean_supported_points": sum(row["clean_supported_points"] for row in rows),
        "represented_clean_points_supported": sum(
            row["represented_clean_points_supported"] for row in rows),
    }
    return result


def prepare_meta_batch(raw: dict, device: str) -> dict:
    values = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor)
              else value for key, value in raw.items()}
    values["radar_sites"] = SparseSites._from_validated(
        values.pop("radar_coords").long(), values.pop("radar_features"),
        GRID.scale_shape(8), 8)
    values["lidar_sites"] = SparseSites._from_validated(
        values.pop("lidar_coords").long(), values.pop("lidar_features"),
        GRID.scale_shape(8), 8)
    values["neighbors"] = Neighborhood(values.pop("neighbor_indices"),
                                        values.pop("neighbor_distances"))
    values["candidate_coords"] = values["candidate_coords"].long()
    values["observed_coords"] = values["observed_coords"].long()
    values["targets"] = CachedOccupancyTargets(
        values.pop("occupied"), values.pop("known_free"),
        values.pop("represented_clean_points_supported"))
    values["clean_supported_voxels"] = len(values["observed_coords"])
    return values
