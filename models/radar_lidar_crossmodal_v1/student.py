"""Radar-only raw sparse confidence and distilled bottleneck features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage1.sparse import SparseSites, encode_keys

from . import INTERFACE_VERSION
from .encoder import BOTTLENECK_CHANNELS, GRID, FrozenOriginalPFE
from .teacher import ResidualAdapter

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None


@dataclass(frozen=True)
class StudentConfig:
    feature_dim: int = BOTTLENECK_CHANNELS
    decoder_channels: tuple[int, int, int] = (32, 16, 16)
    positive_weight: float = 2.0
    dice_weight: float = 0.5
    distillation_weight: float = 1.0
    spatial_weight: float = 1.0
    smooth_l1_beta: float = 1.0
    confidence_threshold: float = 0.5
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def __post_init__(self):
        if self.feature_dim != BOTTLENECK_CHANNELS or min(self.decoder_channels) < 1:
            raise ValueError("Student must use the exact 64-channel original PFE bottleneck")
        if min(self.positive_weight, self.smooth_l1_beta) <= 0 or min(
                self.dice_weight, self.distillation_weight, self.spatial_weight) < 0:
            raise ValueError("Student loss weights are invalid")
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("Confidence threshold must be in (0,1)")
        if (not self.thresholds or any(not 0 < value < 1 for value in self.thresholds)
                or sorted(set(self.thresholds)) != list(self.thresholds)):
            raise ValueError("Validation thresholds must be unique, ascending, and in (0,1)")


class GenerativeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if ME is None:
            raise RuntimeError("MinkowskiEngine is required for unpruned generative sparse decoding")
        self.up = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels, out_channels, kernel_size=2, stride=2, dimension=3)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, tensor):
        tensor = self.up(tensor)
        return ME.SparseTensor(F.silu(self.norm(tensor.F)),
                               coordinate_map_key=tensor.coordinate_map_key,
                               coordinate_manager=tensor.coordinate_manager)


class RawSparseConfidenceDecoder(nn.Module):
    """Expand every bottleneck site by 2×2×2 at each of three levels."""

    def __init__(self, config: StudentConfig):
        super().__init__()
        widths = (config.feature_dim, *config.decoder_channels)
        self.blocks = nn.ModuleList(GenerativeBlock(widths[i], widths[i + 1])
                                    for i in range(3))
        self.logits = nn.Linear(widths[-1], 1)

    def forward(self, bottleneck: SparseSites):
        if bottleneck.stride != 8 or bottleneck.shape_zyx != GRID.scale_shape(8):
            raise ValueError("Expected original PFE stride-8 bottleneck")
        if not len(bottleneck.coords):
            return SparseSites._from_validated(
                bottleneck.coords.new_empty((0, 4)),
                bottleneck.features.new_empty((0, 1)), GRID.shape_zyx, 1)
        # ME stores coordinates in the fine-grid convention at their tensor
        # stride. SparseSites stores compact coarse-cell indices instead.
        fine_coordinates = bottleneck.coords.int().clone()
        fine_coordinates[:, 1:] *= bottleneck.stride
        x = ME.SparseTensor(bottleneck.features.contiguous(),
                            coordinates=fine_coordinates.contiguous(),
                            tensor_stride=bottleneck.stride)
        for block in self.blocks:
            x = block(x)
        coordinates = x.C.to(device=bottleneck.features.device, dtype=torch.long)
        if tuple(int(value) for value in x.tensor_stride) != (1, 1, 1):
            raise RuntimeError("Generative decoder did not reach fine-grid stride 1")
        limits = coordinates.new_tensor(GRID.shape_zyx)
        if ((coordinates[:, 1:] < 0) | (coordinates[:, 1:] >= limits)).any():
            raise RuntimeError("Generative decoder emitted coordinates outside the original grid")
        order = torch.argsort(encode_keys(coordinates, GRID.shape_zyx), stable=True)
        return SparseSites._from_validated(coordinates[order], self.logits(x.F)[order],
                                           GRID.shape_zyx, 1)


@dataclass
class StudentOutput:
    conditioning: SparseSites
    confidence_logits: SparseSites

    def exported(self):
        return {
            "conditioning_features": self.conditioning.features,
            "conditioning_coords": self.conditioning.coords,
            "confidence_logits": self.confidence_logits.features,
            "confidence_probabilities": torch.sigmoid(self.confidence_logits.features),
            "confidence_coords": self.confidence_logits.coords,
            "grid_metadata": {
                "origin_xyz_m": GRID.minimum_xyz,
                "voxel_xyz_m": GRID.size_xyz,
                "conditioning_voxel_xyz_m": tuple(value * 8 for value in GRID.size_xyz),
                "shape_zyx": GRID.shape_zyx,
                "conditioning_stride": 8,
                "confidence_stride": 1,
                "coordinate_order": "batch,z,y,x",
            },
            "interface_version": INTERFACE_VERSION,
        }


class RadarOnlyStudent(nn.Module):
    def __init__(self, radar_checkpoint: str, config: StudentConfig | None = None):
        super().__init__()
        self.config = config or StudentConfig()
        self.radar_encoder = FrozenOriginalPFE(radar_checkpoint, modality="radar")
        d = self.config.feature_dim
        self.shared = ResidualAdapter(d)
        self.feature_head = ResidualAdapter(d)
        self.feature_norm = nn.LayerNorm(d)
        self.confidence_decoder = RawSparseConfidenceDecoder(self.config)

    def train(self, mode: bool = True):
        super().train(mode)
        self.radar_encoder.eval()
        return self

    def forward(self, radar_points, radar_valid) -> StudentOutput:
        # No LiDAR, teacher, LiDAR coordinates or target labels enter this path.
        with torch.no_grad():
            radar = self.radar_encoder(radar_points, radar_valid)
        shared = radar.replace_features(self.shared(radar.features))
        feature = shared.replace_features(self.feature_norm(self.feature_head(shared.features)))
        confidence = self.confidence_decoder(shared)
        return StudentOutput(feature, confidence)
