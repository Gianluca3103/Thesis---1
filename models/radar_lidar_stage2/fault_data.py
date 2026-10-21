"""Faulty LiDAR input for Stage II; real files or reproducible synthetic faults."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


def synthetic_fault(clean: torch.Tensor, frame_id: str, epoch: int,
                    *, seed: int = 3103, dropout: float = 0.25,
                    sector_deg: float = 35.0) -> torch.Tensor:
    """Remove returns in a random azimuth sector, then randomly drop returns."""
    if not 0 <= dropout < 1 or not 0 <= sector_deg < 360:
        raise ValueError("Invalid synthetic LiDAR fault parameters")
    digest = hashlib.sha256(f"{seed}:{frame_id}:{epoch}".encode()).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
    angle = torch.atan2(clean[:, 1], clean[:, 0]).cpu()
    center = (torch.rand((), generator=generator) * 2 - 1) * torch.pi
    distance = torch.atan2(torch.sin(angle - center), torch.cos(angle - center)).abs()
    keep = (distance >= sector_deg * torch.pi / 360) & (
        torch.rand(len(clean), generator=generator) >= dropout)
    return clean[keep.to(clean.device)]


def load_faulty(root: Path, split: str, frame_id: str) -> torch.Tensor:
    paths = (root / split / f"{frame_id}.bin", root / f"{frame_id}.bin")
    matches = [path for path in paths if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one calibrated four-float LiDAR file for {split}/{frame_id} "
            f"under {root}; found {len(matches)}")
    raw = np.fromfile(matches[0], dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"Faulty LiDAR must have four float32 columns: {matches[0]}")
    points = torch.from_numpy(raw.reshape(-1, 4).copy())
    if not torch.isfinite(points).all():
        raise ValueError(f"Faulty LiDAR contains NaN or Inf: {matches[0]}")
    return points
