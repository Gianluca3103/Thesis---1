"""Earlier Stage-I PFE radar/LiDAR encoders with a 64D interface adapter.

The point encoder and four-level sparse backbone are the earlier Stage-I
architecture: 32 input channels and (16, 32, 64, 128) sparse levels. The
128-to-64 adapter is trained during independent pretraining because the
cross-modal teacher and radar-only student consume 64D bottlenecks.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from models.radar_lidar_stage1.pfe_spconv import PFEPointVoxelEncoder, PFESpconvBackbone
from models.radar_lidar_stage1.sparse import SparseSites, encode_keys

from .encoder import BOTTLENECK_CHANNELS, GRID, masked_bottleneck_sites


BACKBONE_CHANNELS = (16, 32, 64, 128)


def _selected_coarse_sites(coordinates: torch.Tensor, rate: float, salt: int) -> torch.Tensor:
    """Use the same stride-8 mask on coarse sites that masked fine inputs."""
    fine_origin = coordinates.clone()
    fine_origin[:, 1:] *= 8
    return masked_bottleneck_sites(fine_origin, rate, salt)


class PreviousStage1PFE(nn.Module):
    def __init__(self, checkpoint: str | Path | None = None, *,
                 modality: str, frozen: bool = False):
        super().__init__()
        if modality not in ("radar", "lidar"):
            raise ValueError("Encoder modality must be radar or lidar")
        if frozen and checkpoint is None:
            raise ValueError("A frozen encoder requires pretrained weights")
        self.point = PFEPointVoxelEncoder(GRID, modality)
        self.backbone = PFESpconvBackbone(BACKBONE_CHANNELS)
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
                raise FileNotFoundError(f"Previous Stage-I PFE weights missing: {checkpoint}")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(saved, dict) or not saved or not all(
                    isinstance(value, torch.Tensor) for value in saved.values()):
                raise ValueError("Expected a direct PreviousStage1PFE state dictionary")
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


class FrozenPreviousStage1PFE(PreviousStage1PFE):
    def __init__(self, checkpoint: str | Path, *, modality: str):
        super().__init__(checkpoint, modality=modality, frozen=True)
