"""Radar-only bottleneck student trained before any surface decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage1.sparse import SparseSites, encode_keys

from .encoder import BOTTLENECK_CHANNELS
from .previous_encoder import FrozenPreviousStage1Sparse
from .teacher import ResidualAdapter


@dataclass(frozen=True)
class FeatureStudentConfig:
    feature_dim: int = BOTTLENECK_CHANNELS
    smooth_l1_beta: float = 1.0

    def __post_init__(self):
        if self.feature_dim != BOTTLENECK_CHANNELS or self.smooth_l1_beta <= 0:
            raise ValueError("The feature student requires a 64D bottleneck and positive beta")


class RadarOnlyFeatureStudent(nn.Module):
    """Distill the frozen teacher's radar-coordinate fused features from radar alone."""

    def __init__(self, radar_checkpoint: str, config: FeatureStudentConfig | None = None):
        super().__init__()
        self.config = config or FeatureStudentConfig()
        self.radar_encoder = FrozenPreviousStage1Sparse(radar_checkpoint, modality="radar")
        self.shared = ResidualAdapter(self.config.feature_dim)
        self.feature_head = ResidualAdapter(self.config.feature_dim)
        self.feature_norm = nn.LayerNorm(self.config.feature_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.radar_encoder.eval()
        return self

    def forward(self, radar_points: torch.Tensor, radar_valid: torch.Tensor) -> SparseSites:
        with torch.no_grad():
            radar = self.radar_encoder(radar_points, radar_valid)
        shared = self.shared(radar.features)
        return radar.replace_features(self.feature_norm(self.feature_head(shared)))


def feature_distillation_loss(student: SparseSites, teacher: SparseSites,
                              config: FeatureStudentConfig) -> torch.Tensor:
    if (student.stride != teacher.stride or student.shape_zyx != teacher.shape_zyx
            or student.features.shape != teacher.features.shape
            or not torch.equal(encode_keys(student.coords, student.shape_zyx),
                               encode_keys(teacher.coords, teacher.shape_zyx))):
        raise ValueError("Teacher and radar-only student bottleneck sites differ")
    if not len(student.coords):
        return student.features.sum() * 0
    return F.smooth_l1_loss(student.features, teacher.features.detach(),
                            beta=config.smooth_l1_beta)
