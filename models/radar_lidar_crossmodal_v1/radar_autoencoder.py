"""Direct radar encoder -> occupancy decoder; clean LiDAR is supervision only.

This ablation keeps the current voxel grid, generative decoder, radar support,
and LiDAR occupancy target definitions. No cross-attention, LiDAR encoder,
teacher features, or distillation module enters its forward or training path.
"""

from __future__ import annotations

import torch
from torch import nn

from .previous_encoder import PreviousStage1Sparse
from .student import RawSparseConfidenceDecoder, StudentConfig, StudentOutput


class RadarOccupancyAutoencoder(nn.Module):
    """Trainable pretrained radar backbone directly decoded to occupancy."""

    def __init__(self, radar_checkpoint: str | None = None,
                 config: StudentConfig | None = None):
        super().__init__()
        self.config = config or StudentConfig(positive_weight=1, dice_weight=0,
                                             lambda_teacher_evidence=0)
        self.radar_encoder = PreviousStage1Sparse(radar_checkpoint, modality="radar")
        self.confidence_decoder = RawSparseConfidenceDecoder(self.config)

    def forward(self, radar_points: torch.Tensor,
                radar_valid: torch.Tensor) -> StudentOutput:
        encoded = self.radar_encoder(radar_points, radar_valid)
        return StudentOutput(encoded, self.confidence_decoder(encoded))
