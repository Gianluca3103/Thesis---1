"""Immutable geometry and labels for meta training with live encoders."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import torch

from models.radar_lidar_stage1.correspondence import Neighborhood

from .encoder import GRID
from .pipeline_data import collate_stage1


VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hash() -> str:
    repo = Path(__file__).resolve().parents[2]
    files = (
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
        raise ValueError("Invalid geometry cache split or neighborhood")
    return {
        "version": VERSION, "split": split, "frame_ids": frame_ids,
        "vod_root": str(vod_root.resolve()), "radar_variant": radar_variant,
        "radar_initial_sha256": _sha256(radar_encoder),
        "lidar_initial_sha256": _sha256(lidar_encoder),
        "radius_m": radius_m, "max_candidates": max_candidates,
        "source_hash": _source_hash(),
        "grid_minimum_xyz": list(GRID.minimum_xyz),
        "grid_size_xyz": list(GRID.size_xyz),
        "grid_shape_zyx": list(GRID.shape_zyx),
    }


class MetaOccupancyCache:
    """Resumable geometry cache; encoder outputs must never appear in entries."""

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
            raise ValueError(f"Geometry cache incompatible; use a new cache root: {manifest}")

    def path(self, frame_id: str) -> Path:
        if frame_id not in self.frames or not frame_id.isdigit():
            raise ValueError("Frame is not in this cache split")
        return self.root / f"{frame_id}.pt"

    def check_complete(self) -> None:
        missing = [frame_id for frame_id in self.frames if not self.path(frame_id).is_file()]
        if missing:
            raise FileNotFoundError(f"Geometry cache lacks {len(missing)} frames; first: {missing[0]}")

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
            raise ValueError(f"Invalid geometry cache identity: {frame_id}")
        if "radar_features" in entry or "lidar_features" in entry:
            raise ValueError("Trainable encoder features cannot be cached")
        radar, lidar, candidate = (entry[f"{name}_coords"]
                                   for name in ("radar", "lidar", "candidate"))
        observed = entry["observed_coords"]
        if (any(value.ndim != 2 or value.shape[1] != 4
                for value in (radar, lidar, candidate, observed))
                or any(bool(value[:, 0].ne(0).any())
                       for value in (radar, lidar, candidate, observed))
                or entry["neighbor_indices"].shape != (len(radar), self.expected["max_candidates"])
                or entry["neighbor_distances"].shape != entry["neighbor_indices"].shape
                or bool(((entry["neighbor_indices"] < -1)
                         | (entry["neighbor_indices"] >= len(lidar))).any())
                or any(entry[name].shape != (len(candidate),)
                       for name in ("occupied", "known_free", "support"))
                or any(entry[name].dtype != torch.bool
                       for name in ("occupied", "known_free", "support"))
                or bool((entry["occupied"] & entry["known_free"]).any())):
            raise ValueError(f"Invalid geometry cache shapes: {frame_id}")

    def _read(self, frame_id: str) -> dict:
        entry = torch.load(self.path(frame_id), map_location="cpu", weights_only=True)
        self._check(entry, frame_id)
        return entry

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self._read(self.frames[index])


class MetaOccupancyDataset:
    """Join geometry to raw point clouds supplied by the fixed-frame cache."""

    def __init__(self, points, geometry: MetaOccupancyCache):
        if [frame.frame_id for frame in points.frames] != geometry.frames:
            raise ValueError("Point and geometry cache frame order differs")
        self.points = points
        self.geometry = geometry
        self.frames = points.frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        row = self.points[index]
        entry = self.geometry[index]
        if row["frame_id"] != entry["frame_id"]:
            raise ValueError("Point and geometry cache frame differs")
        return {**row, "geometry": entry}

    def set_epoch(self, epoch):
        self.points.set_epoch(epoch)


@dataclass(frozen=True)
class CachedOccupancyTargets:
    occupied: torch.Tensor
    known_free: torch.Tensor
    represented_clean_points_supported: int


def collate_meta_cache(rows: list[dict]) -> dict:
    if not rows or any("geometry" not in row for row in rows):
        raise ValueError("Expected point-cloud rows joined with geometry")
    result = collate_stage1(rows)
    entries = [row["geometry"] for row in rows]

    def coordinates(name):
        return torch.cat([
            torch.cat((torch.full_like(entry[name][:, :1], batch), entry[name][:, 1:]), dim=1)
            for batch, entry in enumerate(entries)
        ])

    offsets, offset = [], 0
    for entry in entries:
        local = entry["neighbor_indices"].long()
        offsets.append(torch.where(local >= 0, local + offset, local))
        offset += len(entry["lidar_coords"])
    result.update({
        "radar_coords": coordinates("radar_coords"),
        "lidar_coords": coordinates("lidar_coords"),
        "neighbor_indices": torch.cat(offsets),
        "neighbor_distances": torch.cat([entry["neighbor_distances"] for entry in entries]),
        "candidate_coords": coordinates("candidate_coords"),
        "observed_coords": coordinates("observed_coords"),
        "occupied": torch.cat([entry["occupied"] for entry in entries]),
        "known_free": torch.cat([entry["known_free"] for entry in entries]),
        "support": torch.cat([entry["support"] for entry in entries]),
        "clean_supported_points": sum(entry["clean_supported_points"] for entry in entries),
        "represented_clean_points_supported": sum(
            entry["represented_clean_points_supported"] for entry in entries),
    })
    return result


def prepare_meta_batch(raw: dict, device: str) -> dict:
    values = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor)
              else value for key, value in raw.items()}
    for name in ("radar_coords", "lidar_coords", "candidate_coords", "observed_coords"):
        values[name] = values[name].long()
    values["neighbors"] = Neighborhood(values.pop("neighbor_indices"),
                                        values.pop("neighbor_distances"))
    values["targets"] = CachedOccupancyTargets(
        values.pop("occupied"), values.pop("known_free"),
        values.pop("represented_clean_points_supported"))
    values["clean_supported_voxels"] = len(values["observed_coords"])
    return values
