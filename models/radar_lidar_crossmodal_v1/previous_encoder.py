"""Native sparse backbone used by the earlier Stage-I interpolation views.

The point encoder starts at 16 channels and the four sparse levels have
widths (16, 32, 64, 128). The final level grows its active support axially,
as in that Stage-I model. An independently trained adapter maps its stride-8
features to the existing 64D cross-modal interface.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from models.radar_lidar_stage1.sparse import (
    PointVoxelEncoder, SparseBackbone, SparseSites, encode_keys,
)

from .encoder import BOTTLENECK_CHANNELS, GRID, masked_bottleneck_sites


BACKBONE_CHANNELS = (16, 32, 64, 128)
GROWTH_SCALES = (4,)


def _selected_coarse_sites(coordinates: torch.Tensor, rate: float, salt: int) -> torch.Tensor:
    """Use the same stride-8 mask on coarse sites that masked fine inputs."""
    fine_origin = coordinates.clone()
    fine_origin[:, 1:] *= 8
    return masked_bottleneck_sites(fine_origin, rate, salt)


class PreviousStage1Sparse(nn.Module):
    def __init__(self, checkpoint: str | Path | None = None, *,
                 modality: str, frozen: bool = False):
        super().__init__()
        if modality not in ("radar", "lidar"):
            raise ValueError("Encoder modality must be radar or lidar")
        if frozen and checkpoint is None:
            raise ValueError("A frozen encoder requires pretrained weights")
        self.point = PointVoxelEncoder(GRID, modality, BACKBONE_CHANNELS[0])
        self.backbone = SparseBackbone(BACKBONE_CHANNELS, GROWTH_SCALES)
        self.output_adapter = nn.Sequential(
            nn.Linear(BACKBONE_CHANNELS[-1], BOTTLENECK_CHANNELS),
            nn.LayerNorm(BOTTLENECK_CHANNELS), nn.SiLU(),
            nn.Linear(BOTTLENECK_CHANNELS, BOTTLENECK_CHANNELS),
        )
        self.modality = modality
        self.frozen = frozen
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Previous Stage-I sparse weights missing: {checkpoint}")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(saved, dict) or not saved or not all(
                    isinstance(value, torch.Tensor) for value in saved.values()):
                raise ValueError("Expected a direct PreviousStage1Sparse state dictionary")
            self.load_state_dict(saved, strict=True)
        self.source_checkpoint = str(checkpoint.resolve()) if checkpoint is not None else None
        if frozen:
            self.requires_grad_(False)
            self.eval()

    def train(self, mode: bool = True):
        super().train(False if self.frozen else mode)
        return self

    def forward(self, points: torch.Tensor, valid: torch.Tensor, *,
                input_mask_rate: float = 0.0, mask_salt: int = 0) -> SparseSites:
        if self.frozen:
            if input_mask_rate:
                raise ValueError("Frozen encoder cannot mask input features")
            with torch.no_grad():
                return self._encode(points, valid, 0.0, mask_salt)
        return self._encode(points, valid, input_mask_rate, mask_salt)

    def _encode(self, points: torch.Tensor, valid: torch.Tensor,
                input_mask_rate: float, mask_salt: int) -> SparseSites:
        fine, _ = self.point(points, valid, defer_diagnostics=True)
        if input_mask_rate:
            masked = masked_bottleneck_sites(fine.coords, input_mask_rate, mask_salt)
            fine = fine.replace_features(fine.features.masked_fill(masked[:, None], 0.0))
        levels, _ = self.backbone(fine, defer_diagnostics=True)
        coarse = levels[-1]
        if not len(coarse.coords):
            return SparseSites._from_validated(
                coarse.coords, coarse.features.new_empty((0, BOTTLENECK_CHANNELS)),
                GRID.scale_shape(8), 8)
        order = torch.argsort(encode_keys(coarse.coords, GRID.scale_shape(8)), stable=True)
        return SparseSites._from_validated(
            coarse.coords[order], self.output_adapter(coarse.features[order]),
            GRID.scale_shape(8), 8)


class FrozenPreviousStage1Sparse(PreviousStage1Sparse):
    def __init__(self, checkpoint: str | Path, *, modality: str):
        super().__init__(checkpoint, modality=modality, frozen=True)
