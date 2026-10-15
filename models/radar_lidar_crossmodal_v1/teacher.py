"""Frozen previous Stage-I encoders with an independently trainable teacher."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage1.correspondence import Neighborhood, local_neighbors
from models.radar_lidar_stage1.sparse import SparseSites

from .encoder import BOTTLENECK_CHANNELS, GRID
from .previous_encoder import FrozenPreviousStage1Sparse


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
            raise ValueError("The encoder interface has 64 channels and uses four 16D heads")
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
            raise ValueError("Expected previous Stage-I stride-8 sparse bottlenecks")
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
        self.radar_encoder = FrozenPreviousStage1Sparse(radar_checkpoint, modality="radar")
        self.lidar_encoder = FrozenPreviousStage1Sparse(lidar_checkpoint, modality="lidar")
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
            if radar.coords.is_cuda and os.environ.get("CROSSMODAL_GPU_NEIGHBORS") == "1":
                neighbors = tiled_local_neighbors(radar, lidar, self.config.radius_m,
                                                  self.config.max_candidates)
            else:
                neighbors = local_neighbors(radar, lidar, GRID, self.config.radius_m,
                                            self.config.max_candidates)
        return self.core(radar, lidar, neighbors)


@torch.no_grad()
def tiled_local_neighbors(radar: SparseSites, lidar: SparseSites,
                          radius_m: float, max_neighbors: int,
                          *, max_pairs: int = 2_000_000) -> Neighborhood:
    """Exact top-k radius search in bounded GPU tiles, preserving batch IDs.

    This optional path removes the GPU→CPU→GPU KD-tree transfer. Its temporary
    distance matrix is limited by max_pairs; the CPU KD-tree remains available.
    """
    if radar.stride != lidar.stride or radius_m <= 0 or max_neighbors < 1 or max_pairs < 1:
        raise ValueError("Incompatible local neighborhood settings")
    if radar.coords.device != lidar.coords.device:
        raise ValueError("Radar and LiDAR sites must share a device")
    n, m = len(radar.coords), len(lidar.coords)
    indices = torch.full((n, max_neighbors), -1, dtype=torch.long,
                         device=radar.coords.device)
    distances = radar.features.new_full((n, max_neighbors), float("inf"))
    if not n or not m:
        return Neighborhood(indices, distances)
    query = radar.centers_xyz(GRID)
    reference = lidar.centers_xyz(GRID)
    k = min(max_neighbors, m)
    tile = max(1, max_pairs // m)
    for start in range(0, n, tile):
        end = min(n, start + tile)
        matrix = torch.cdist(query[start:end], reference)
        same_batch = radar.coords[start:end, :1] == lidar.coords[:, 0][None, :]
        matrix.masked_fill_(~same_batch, float("inf"))
        nearest_distance, nearest_index = matrix.topk(k, dim=1, largest=False, sorted=True)
        valid = nearest_distance <= radius_m
        indices[start:end, :k] = nearest_index.masked_fill(~valid, -1)
        distances[start:end, :k] = nearest_distance.masked_fill(~valid, float("inf"))
    return Neighborhood(indices, distances)
