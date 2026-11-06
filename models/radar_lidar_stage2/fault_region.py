"""Synthetic injector footprint for the mild-fault Stage-II baseline.

This is an oracle sidecar derived from the injector's clean input and seed.
It is available for synthetic experiments, not for unknown real faults.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from .sparse_geometry import GRID


VERSION = "mild_fault_injector_azimuth_footprint_v1"


def partial_azimuth_region(clean_xyz: np.ndarray, remove_fraction: float,
                           seed: int) -> dict:
    """Sector of a deterministic partial-loss injector, with boundary midpoints."""
    points = np.asarray(clean_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Clean LiDAR must be finite XYZ")
    if remove_fraction not in (0.30, 0.40, 0.50):
        raise ValueError("Unsupported partial-loss fraction")
    count = len(points)
    removed_count = min(int(np.floor(remove_fraction * count + 0.5)), count // 2)
    if not removed_count:
        return {"kind": "empty", "removed_points": 0}
    angles = np.arctan2(points[:, 1], points[:, 0])
    order = np.argsort(angles, kind="stable")
    start = int(np.random.default_rng(seed).integers(count))
    before = float(angles[order[(start - 1) % count]])
    first = float(angles[order[start]])
    last = float(angles[order[(start + removed_count - 1) % count]])
    after = float(angles[order[(start + removed_count) % count]])

    def midpoint_ccw(a: float, b: float) -> float:
        return (a + ((b - a) % math.tau) * 0.5) % math.tau

    return {"kind": "azimuth_sector",
            "start_rad": midpoint_ccw(before, first),
            "end_rad": midpoint_ccw(last, after),
            "removed_points": removed_count}


def footprint_mask(fine_bzyx: torch.Tensor, region: dict) -> torch.Tensor:
    """Candidate voxel intersects the injector sector (one-cell boundary halo)."""
    kind = region.get("kind")
    if kind == "full_scan":
        return torch.ones(len(fine_bzyx), dtype=torch.bool,
                          device=fine_bzyx.device)
    if kind == "empty":
        return torch.zeros(len(fine_bzyx), dtype=torch.bool,
                           device=fine_bzyx.device)
    if kind != "azimuth_sector":
        raise ValueError(f"Unknown injector footprint: {kind}")
    start = float(region["start_rad"])
    end = float(region["end_rad"])
    if not all(math.isfinite(v) and 0 <= v < math.tau for v in (start, end)):
        raise ValueError("Invalid injector azimuth bounds")
    if fine_bzyx.ndim != 2 or fine_bzyx.shape[1] != 4:
        raise ValueError("Injector footprint requires [batch,z,y,x] cells")
    centers = (fine_bzyx[:, [3, 2]].to(torch.float64) + 0.5)
    centers = centers * centers.new_tensor(GRID.voxel_xyz[:2])
    centers += centers.new_tensor(GRID.minimum_xyz[:2])
    angles = torch.atan2(centers[:, 1], centers[:, 0])
    relative = torch.remainder(angles - start, math.tau)
    width = (end - start) % math.tau
    radius = torch.linalg.vector_norm(centers, dim=1)
    half_diagonal = math.hypot(GRID.voxel_xyz[0], GRID.voxel_xyz[1]) * 0.5
    margin = torch.asin((half_diagonal / radius.clamp_min(half_diagonal))
                        .clamp(max=1.0))
    return (relative <= width + margin) | (relative >= math.tau - margin)


def cache_identity(split: str, frame_ids: list[str], mild_manifest: Path,
                   fixed_manifest: Path) -> dict:
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"version": VERSION, "split": split, "frame_ids": frame_ids,
            "source_sha256": source,
            "mild_manifest_sha256": hashlib.sha256(mild_manifest.read_bytes()).hexdigest(),
            "fixed_manifest_sha256": hashlib.sha256(fixed_manifest.read_bytes()).hexdigest()}


class FaultRegionCache:
    def __init__(self, root: Path, expected: dict, *, create: bool = False):
        self.root = Path(root)
        manifest = self.root / "manifest.json"
        if create and not manifest.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        if json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Fault-region cache policy differs: {manifest}")
        self.regions_path = self.root / "regions.json"
        self.regions = (json.loads(self.regions_path.read_text(encoding="utf-8"))
                        if self.regions_path.exists() else None)
        if self.regions is not None and set(self.regions) != set(expected["frame_ids"]):
            raise ValueError(f"Incomplete fault-region cache: {self.regions_path}")

    def write(self, regions: dict[str, dict]) -> None:
        if set(regions) != set(json.loads((self.root / "manifest.json")
                                        .read_text(encoding="utf-8"))["frame_ids"]):
            raise ValueError("Fault-region cache must contain every frame")
        temporary = self.regions_path.with_name(
            self.regions_path.name + f".{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(regions, separators=(",", ":")),
                                 encoding="utf-8")
            temporary.replace(self.regions_path)
        finally:
            temporary.unlink(missing_ok=True)
        self.regions = regions

    def get(self, frame_id: str) -> dict:
        if self.regions is None:
            raise FileNotFoundError(f"Fault-region cache not built: {self.regions_path}")
        return self.regions[frame_id]
