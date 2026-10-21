"""Frozen Stage-I interface and Stage-II checkpoint loading."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import torch

from models.radar_lidar_crossmodal_v1.meta_occupancy import radar_only_probability_map
from models.radar_lidar_crossmodal_v1.train_radar_autoencoder import load_radar_autoencoder

from .sparse_geometry import GRID
from .sparse_residual import VERSION, SparseResidualStage2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage1(path: Path, device: str, radar_variant: str):
    model, saved = load_radar_autoencoder(path, device)
    if saved.get("radar_variant") != radar_variant:
        raise ValueError("Stage-I radar variant does not match the paired dataset")
    model.requires_grad_(False)
    return model, saved


@torch.no_grad()
def stage1_frame(model, radar: torch.Tensor, radius_m: float) -> dict:
    if radar.ndim != 2 or radar.shape[1] != 7:
        raise ValueError("Stage I expects calibrated seven-column radar points")
    batch = radar[None]
    valid = torch.ones((1, len(radar)), device=radar.device, dtype=torch.bool)
    output = radar_only_probability_map(model, batch, valid, radius_m)
    if output["grid_metadata"]["voxel_xyz_m"] != GRID.voxel_xyz:
        raise ValueError("Stage-I and Stage-II fine voxel sizes differ")
    return output


def load_stage2(path: Path, device: str,
                *, phase: str = "diffusion") -> tuple[SparseResidualStage2, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("version") != VERSION or saved.get("phase") != phase:
        raise ValueError(f"Expected {VERSION} {phase} checkpoint")
    if saved.get("grid") != asdict(GRID):
        raise ValueError("Stage-II checkpoint grid differs from current code")
    model = SparseResidualStage2().to(device)
    model.load_state_dict(saved["model"], strict=True)
    return model.eval(), saved
