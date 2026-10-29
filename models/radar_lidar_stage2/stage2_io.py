"""Frozen Stage-I interface and Stage-II checkpoint loading."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import torch

from models.radar_lidar_crossmodal_v1.meta_occupancy import radar_only_probability_map
from models.radar_lidar_crossmodal_v1.train_radar_autoencoder import load_radar_autoencoder
from models.radar_lidar_crossmodal_v1.train_student import load_student

from .sparse_geometry import GRID
from .sparse_residual import LPCR_REVISION, MODEL_REVISION, VERSION, SparseResidualStage2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage1(path: Path, device: str, radar_variant: str):
    header = torch.load(path, map_location="cpu", weights_only=False)
    if header.get("version") == "crossmodal_student_v5_multiscale_refined_raw64":
        model, saved = load_student(path, device)
    else:
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
    previous_diffusion_revision = "stage2_r2ldm_centroid_residual_v8"
    accepted_revisions = ({LPCR_REVISION} if phase == "lpcr"
                          else {MODEL_REVISION, LPCR_REVISION})
    if phase == "diffusion":
        accepted_revisions.add(previous_diffusion_revision)
    if (saved.get("version") != VERSION
            or saved.get("model_revision") not in accepted_revisions
            or saved.get("phase") != phase):
        raise ValueError(f"Expected {VERSION}/{accepted_revisions} {phase} checkpoint")
    if saved.get("grid") != asdict(GRID):
        raise ValueError("Stage-II checkpoint grid differs from current code")
    model = SparseResidualStage2().to(device)
    missing, unexpected = model.load_state_dict(saved["model"], strict=False)
    if (unexpected or
            (saved["model_revision"] == MODEL_REVISION and missing) or
            any(not key.startswith("addition_head.") for key in missing)):
        raise ValueError("Stage-II checkpoint model weights differ from the architecture")
    model.use_addition_head = saved["model_revision"] == MODEL_REVISION
    model.lpcr_uses_stage1_probabilities = (
        saved["model_revision"] == previous_diffusion_revision)
    return model.eval(), saved
