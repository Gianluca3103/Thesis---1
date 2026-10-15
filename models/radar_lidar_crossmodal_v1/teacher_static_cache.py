"""Frozen encoder bottlenecks, neighbors, and object labels for an unaugmented teacher."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from models.radar_lidar_stage1.correspondence import Neighborhood
from models.radar_lidar_stage1.sparse import SparseSites

from .encoder import GRID


VERSION = 1


def _source_hash() -> str:
    repo = Path(__file__).resolve().parents[2]
    files = (Path(__file__), repo / "models/radar_lidar_stage1/correspondence.py",
             repo / "models/radar_lidar_stage1/sparse.py",
             repo / "models/radar_lidar_crossmodal_v1/teacher.py",
             repo / "models/radar_lidar_crossmodal_v1/supervision.py",
             repo / "models/radar_lidar_crossmodal_v1/previous_encoder.py")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


class TeacherStaticCache:
    def __init__(self, root: Path, *, split: str, radar_variant: str,
                 radar_sha256: str, lidar_sha256: str, radius_m: float,
                 max_candidates: int, create: bool = False):
        if split not in ("train", "val"):
            raise ValueError("Teacher cache split must be train or val")
        self.root = Path(root) / split
        expected = {
            "version": VERSION, "split": split, "radar_variant": radar_variant,
            "radar_sha256": radar_sha256, "lidar_sha256": lidar_sha256,
            "radius_m": radius_m, "max_candidates": max_candidates,
            "source_hash": _source_hash(), "augmentation_enabled": False,
            "grid_minimum_xyz": list(GRID.minimum_xyz),
            "grid_size_xyz": list(GRID.size_xyz),
        }
        manifest = self.root / "manifest.json"
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            if not manifest.exists():
                temporary = manifest.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(expected), encoding="utf-8")
                temporary.replace(manifest)
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise ValueError("Teacher cache manifest differs from encoders, geometry, or source")

    def path(self, frame_id: str) -> Path:
        if not frame_id.isdigit():
            raise ValueError("Expected numeric VoD frame ID")
        return self.root / f"{frame_id}.pt"

    def write(self, frame_id: str, radar: SparseSites, lidar: SparseSites,
              neighbors: Neighborhood, radar_ids: torch.Tensor,
              lidar_ids: torch.Tensor, classes: dict[int, str]) -> int:
        path = self.path(frame_id)
        if path.exists():
            self._read(frame_id)
            return path.stat().st_size
        if (radar.coords[:, 0].ne(0).any() or lidar.coords[:, 0].ne(0).any()
                or radar_ids.shape != (len(radar.coords),)
                or lidar_ids.shape != (len(lidar.coords),)
                or neighbors.indices.shape[0] != len(radar.coords)):
            raise ValueError("Teacher cache entries must be aligned single frames")
        entry = {
            "version": VERSION, "frame_id": frame_id,
            "radar_coords": radar.coords.to("cpu", dtype=torch.int32),
            "radar_features": radar.features.detach().cpu(),
            "lidar_coords": lidar.coords.to("cpu", dtype=torch.int32),
            "lidar_features": lidar.features.detach().cpu(),
            "neighbor_indices": neighbors.indices.to("cpu", dtype=torch.int32),
            "neighbor_distances": neighbors.distances_m.cpu(),
            "radar_ids": radar_ids.cpu(), "lidar_ids": lidar_ids.cpu(),
            "classes": classes,
        }
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            torch.save(entry, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path.stat().st_size

    def _read(self, frame_id: str) -> dict:
        entry = torch.load(self.path(frame_id), map_location="cpu", weights_only=True)
        if entry.get("version") != VERSION or entry.get("frame_id") != frame_id:
            raise ValueError(f"Invalid teacher cache entry: {frame_id}")
        r, l = entry["radar_coords"], entry["lidar_coords"]
        if (r.ndim != 2 or r.shape[1] != 4 or l.ndim != 2 or l.shape[1] != 4
                or r[:, 0].ne(0).any() or l[:, 0].ne(0).any()
                or entry["radar_features"].shape != (len(r), 64)
                or entry["lidar_features"].shape != (len(l), 64)
                or entry["radar_ids"].shape != (len(r),)
                or entry["lidar_ids"].shape != (len(l),)
                or entry["neighbor_indices"].shape[0] != len(r)
                or entry["neighbor_distances"].shape != entry["neighbor_indices"].shape):
            raise ValueError(f"Invalid teacher cache shapes: {frame_id}")
        return entry

    def load_batch(self, frame_ids: list[str], device: str):
        entries = [self._read(frame_id) for frame_id in frame_ids]

        def sites(prefix):
            coords = torch.cat([
                torch.cat((entry[f"{prefix}_coords"][:, :1] + batch,
                           entry[f"{prefix}_coords"][:, 1:]), 1)
                for batch, entry in enumerate(entries)
            ]).to(device=device, dtype=torch.long)
            features = torch.cat([entry[f"{prefix}_features"] for entry in entries]).to(device)
            return SparseSites._from_validated(coords, features, GRID.scale_shape(8), 8)

        radar, lidar = sites("radar"), sites("lidar")
        offsets = []
        offset = 0
        for entry in entries:
            local = entry["neighbor_indices"].long()
            offsets.append(torch.where(local >= 0, local + offset, local))
            offset += len(entry["lidar_coords"])
        neighbors = Neighborhood(
            torch.cat(offsets).to(device),
            torch.cat([entry["neighbor_distances"] for entry in entries]).to(device))

        def ids(prefix):
            return torch.cat([
                torch.where(entry[f"{prefix}_ids"] >= 0,
                            entry[f"{prefix}_ids"] + batch * 1_000_000,
                            entry[f"{prefix}_ids"])
                for batch, entry in enumerate(entries)
            ]).to(device)

        classes = {identity + batch * 1_000_000: name
                   for batch, entry in enumerate(entries)
                   for identity, name in entry["classes"].items()}
        return radar, lidar, neighbors, ids("radar"), ids("lidar"), classes
