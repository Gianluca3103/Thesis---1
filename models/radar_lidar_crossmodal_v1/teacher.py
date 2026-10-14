"""Frozen exact-PFE encoders with an independently trainable bottleneck teacher."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage1.correspondence import Neighborhood, local_neighbors
from models.radar_lidar_stage1.sparse import SparseSites

from .encoder import BOTTLENECK_CHANNELS, GRID, FrozenOriginalPFE


@dataclass(frozen=True)
class TeacherConfig:
    feature_dim: int = BOTTLENECK_CHANNELS
    heads: int = 4
    max_candidates: int = 24
    radius_m: float = 6.0
    sigma_m: float = 2.0
    temperature: float = 0.1
    adapters_enabled: bool = True
    fusion_beta: float = 1.0
    object_weight: float = 1.0
    geometry_weight: float = 0.5
    contrastive_weight: float = 0.1
    fusion_weight: float = 0.5

    def __post_init__(self):
        if self.feature_dim != BOTTLENECK_CHANNELS or self.heads != 4 or self.feature_dim % self.heads:
            raise ValueError("The exact Downloads encoder has 64 channels and uses four 16D heads")
        if min(self.max_candidates, self.radius_m, self.sigma_m, self.temperature,
               self.fusion_beta) <= 0:
            raise ValueError("Teacher geometry and loss scales must be positive")
        if min(self.object_weight, self.geometry_weight, self.contrastive_weight,
               self.fusion_weight) < 0:
            raise ValueError("Teacher loss weights must be nonnegative")


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.first = nn.Linear(dim, dim)
        self.second = nn.Linear(dim, dim)
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)

    def forward(self, features):
        return features + self.second(F.silu(self.first(self.norm(features))))


@dataclass
class TeacherOutput:
    fused: SparseSites
    radar_frozen: SparseSites
    lidar_frozen: SparseSites
    neighbors: Neighborhood
    attention: torch.Tensor  # [N, heads, K]
    gate: torch.Tensor  # [N,1]
    lidar_contribution: torch.Tensor  # [N,D]
    radar_alignment: torch.Tensor
    lidar_alignment: torch.Tensor


class BottleneckTeacherCore(nn.Module):
    """CPU-testable trainable module; sparse encoders are supplied separately."""

    def __init__(self, config: TeacherConfig):
        super().__init__()
        self.config = config
        d = config.feature_dim
        self.radar_adapter = ResidualAdapter(d)
        self.lidar_adapter = ResidualAdapter(d)
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.output = nn.Linear(d, d)
        self.gate = nn.Linear(2 * d, 1)
        nn.init.constant_(self.gate.bias, -2.0)
        self.fused_norm = nn.LayerNorm(d)
        self.radar_alignment = nn.Linear(d, d)
        self.lidar_alignment = nn.Linear(d, d)

    def forward(self, radar: SparseSites, lidar: SparseSites,
                neighbors: Neighborhood) -> TeacherOutput:
        if radar.stride != 8 or lidar.stride != 8 or radar.shape_zyx != GRID.scale_shape(8):
            raise ValueError("Expected original PFE stride-8 sparse bottlenecks")
        n, k = neighbors.indices.shape
        d, h = self.config.feature_dim, self.config.heads
        if n != len(radar.coords) or k != self.config.max_candidates:
            raise ValueError("Neighbor indices do not match radar queries or configured candidates")
        hr = self.radar_adapter(radar.features) if self.config.adapters_enabled else radar.features
        hl = self.lidar_adapter(lidar.features) if self.config.adapters_enabled else lidar.features
        q = self.query(hr).reshape(n, h, d // h)
        kk = self.key(hl).reshape(len(hl), h, d // h)
        vv = self.value(hl).reshape(len(hl), h, d // h)
        valid = neighbors.valid
        if len(hl):
            gathered_k = kk[neighbors.indices.clamp_min(0)]
            gathered_v = vv[neighbors.indices.clamp_min(0)]
            logits = (q[:, None] * gathered_k).sum(-1).transpose(1, 2) / math.sqrt(d // h)
            logits = logits.masked_fill(~valid[:, None, :], -1e4)
            attention = torch.softmax(logits, -1) * valid[:, None, :]
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
            context = (attention.transpose(1, 2)[..., None] * gathered_v).sum(1).reshape(n, d)
            contribution = self.output(context) * valid.any(-1)[:, None]
        else:
            attention = radar.features.new_zeros((n, h, k))
            contribution = radar.features.new_zeros((n, d))
        gate = torch.sigmoid(self.gate(torch.cat((radar.features, contribution), -1)))
        fused = self.fused_norm(radar.features + gate * contribution)
        return TeacherOutput(
            radar.replace_features(fused), radar, lidar, neighbors, attention, gate,
            contribution, F.normalize(self.radar_alignment(hr), dim=-1),
            F.normalize(self.lidar_alignment(hl), dim=-1))


class CrossModalTeacher(nn.Module):
    def __init__(self, radar_checkpoint: str, lidar_checkpoint: str,
                 config: TeacherConfig | None = None):
        super().__init__()
        self.config = config or TeacherConfig()
        self.radar_encoder = FrozenOriginalPFE(radar_checkpoint)
        self.lidar_encoder = FrozenOriginalPFE(lidar_checkpoint)
        self.core = BottleneckTeacherCore(self.config)

    def train(self, mode: bool = True):
        super().train(mode)
        self.radar_encoder.eval()
        self.lidar_encoder.eval()
        return self

    def forward(self, radar_points, radar_valid, lidar_points, lidar_valid):
        with torch.no_grad():
            radar = self.radar_encoder(radar_points, radar_valid)
            lidar = self.lidar_encoder(lidar_points, lidar_valid)
            neighbors = local_neighbors(radar, lidar, GRID, self.config.radius_m,
                                        self.config.max_candidates)
        return self.core(radar, lidar, neighbors)
