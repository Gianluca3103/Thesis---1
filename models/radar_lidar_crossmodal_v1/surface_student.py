"""Radar-only surface proposals for the cross-modal student.

The confidence branch predicts locations and logits directly. It never
enumerates the 8x8x8 children of a bottleneck cell. The frozen radar encoder
and distilled 64-channel teacher interface remain the same.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage1.correspondence import local_neighbors_xyz
from models.radar_lidar_stage1.sparse import SparseSites, decode_keys, encode_keys

from .encoder import BOTTLENECK_CHANNELS, GRID
from .previous_encoder import FrozenPreviousStage1Sparse
from .pointer_voxels import merge_surface_pointers
from .teacher import ResidualAdapter


@dataclass(frozen=True)
class SurfaceStudentConfig:
    feature_dim: int = BOTTLENECK_CHANNELS
    proposals_per_anchor: int = 24
    anchor_stride: int = 1
    offset_limit_m: float = 1.5
    context_radii_m: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0)
    context_neighbors: int = 16
    # Retained only to deserialize older surface checkpoints.
    confidence_sigma_m: float = 0.2
    target_radius_m: float = 1.5
    target_neighbors: int = 24
    distillation_weight: float = 0.0
    geometry_weight: float = 0.0
    confidence_weight: float = 0.0
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    hit_weight: float = 0.5
    hit_scale: float = 100.0
    threshold: float = 0.25
    evaluation_tolerance_m: float = 0.2
    decoder_from_conditioned: bool = False  # False reloads earlier surface checkpoints.
    pointer_voxel_mode: bool = True  # Old checkpoints reload with this disabled.
    spherical_offsets: bool = False  # Preserve older per-axis offset checkpoints.

    def __post_init__(self):
        if (self.feature_dim != BOTTLENECK_CHANNELS or self.proposals_per_anchor < 1
                or (self.pointer_voxel_mode and self.proposals_per_anchor > 24)
                or self.anchor_stride not in (1, 2, 4, 8) or self.context_neighbors < 1
                or self.target_neighbors < 1 or len(self.context_radii_m) != 4
                or min(self.context_radii_m) <= 0
                or min(self.offset_limit_m, self.confidence_sigma_m,
                       self.target_radius_m, self.evaluation_tolerance_m) <= 0
                or min(self.distillation_weight, self.geometry_weight,
                       self.confidence_weight, self.bce_weight,
                       self.dice_weight, self.hit_weight) < 0
                or self.hit_scale <= 0 or not 0 < self.threshold < 1):
            raise ValueError("Invalid radar-only surface proposal configuration")


@dataclass
class SurfaceProposals:
    xyz: torch.Tensor  # [anchors, slots, 3], continuous LiDAR-frame metres
    logits: torch.Tensor  # [anchors, slots]
    anchor_xyz: torch.Tensor  # [anchors, 3]
    anchor_batch: torch.Tensor  # [anchors]
    guide: torch.Tensor  # learned radar-only features for optional interpolation
    pointer_voxel_mode: bool = True
    selection_threshold: float = 0.25

    @property
    def score(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)


def pointer_xyz_from_offsets(centers: torch.Tensor, raw_offsets: torch.Tensor,
                             radius_m: float, spherical: bool) -> torch.Tensor:
    direction = torch.tanh(raw_offsets)
    if spherical:
        # The legacy head clipped each axis independently. This norm cap is
        # required for a true Euclidean association radius.
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0)
    return centers[:, None, :] + direction * radius_m


class RadarSurfaceHead(nn.Module):
    """Local radar-pattern attention and continuous surface hypotheses."""

    def __init__(self, config: SurfaceStudentConfig):
        super().__init__()
        self.config = config
        width = 16
        source_widths = (16, 32, 64, config.feature_dim)
        self.query = nn.Linear(config.feature_dim if config.decoder_from_conditioned else width,
                               width)
        self.key = nn.ModuleList(nn.Linear(ch, width) for ch in source_widths)
        self.value = nn.ModuleList(nn.Linear(ch, width) for ch in source_widths)
        self.relative = nn.ModuleList(nn.Linear(3, 1, bias=False) for _ in source_widths)
        self.hidden = nn.Sequential(nn.Linear(width * 5, width * 2),
                                    nn.LayerNorm(width * 2), nn.SiLU())
        # Retained so older surface checkpoints still reload. The trained
        # decoder hidden state itself guides optional edge-aware viewing;
        # this separate projection has no occupancy supervision.
        self.guide = nn.Linear(width * 2, width * 2)
        self.guide.requires_grad_(not config.pointer_voxel_mode)
        self.slots = nn.Linear(width * 2, 4 * config.proposals_per_anchor)

    def forward(self, sources: tuple[SparseSites, SparseSites, SparseSites, SparseSites]
                ) -> SurfaceProposals:
        fine = sources[0]
        if tuple(source.stride for source in sources) != (1, 2, 4, 8):
            raise ValueError("Expected radar features at strides 1, 2, 4, and 8")
        if fine.features.shape[1] != 16:
            raise ValueError("The previous Stage-I fine radar encoder has 16 channels")
        if self.config.anchor_stride == 1:
            # Every active fine radar voxel can propose surfaces. No child
            # lattice or radius-based candidate domain is generated here.
            pooled = fine.features
            centers = fine.centers_xyz(GRID)
            batch = fine.coords[:, 0]
        else:
            coarse = fine.coords.clone()
            coarse[:, 1:] //= self.config.anchor_stride
            shape = GRID.scale_shape(self.config.anchor_stride)
            keys, inverse = torch.unique(encode_keys(coarse, shape),
                                         sorted=True, return_inverse=True)
            count = fine.features.new_zeros((len(keys), 1))
            count.index_add_(0, inverse, fine.features.new_ones((len(inverse), 1)))
            pooled = fine.features.new_zeros((len(keys), 16))
            pooled.index_add_(0, inverse, fine.features)
            pooled = pooled / count.clamp_min(1)
            centers = fine.features.new_zeros((len(keys), 3))
            centers.index_add_(0, inverse, fine.centers_xyz(GRID))
            centers = centers / count.clamp_min(1)
            batch = decode_keys(keys, shape)[:, 0]
        if self.config.decoder_from_conditioned:
            # A fine radar site inherits the distilled feature of its own
            # stride-8 parent. The batch coordinate is part of each key.
            parent = (fine.coords if self.config.anchor_stride == 1 else
                      decode_keys(keys, GRID.scale_shape(self.config.anchor_stride)).clone())
            parent = parent.clone()
            parent[:, 1:] = parent[:, 1:] * self.config.anchor_stride // 8
            shape = GRID.scale_shape(8)
            source_keys = encode_keys(sources[3].coords, shape)
            parent_keys = encode_keys(parent, shape)
            sorted_keys, order = torch.sort(source_keys)
            indices = torch.searchsorted(sorted_keys, parent_keys)
            found = (indices < len(sorted_keys))
            if len(sorted_keys):
                found &= sorted_keys[indices.clamp_max(len(sorted_keys) - 1)] == parent_keys
            if not bool(found.all()):
                raise RuntimeError("A radar surface anchor has no conditioned stride-8 parent")
            query = self.query(sources[3].features[order[indices]])
            parts = [query]
        else:
            query = self.query(pooled)
            parts = [pooled]
        for level, source in enumerate(sources):
            radius = self.config.context_radii_m[level]
            neighborhood = local_neighbors_xyz(
                centers, batch, source, GRID, radius, self.config.context_neighbors)
            valid = neighborhood.valid
            if not len(source.coords):
                parts.append(query.new_zeros(query.shape))
                continue
            selected = neighborhood.indices.clamp_min(0)
            relative = (source.centers_xyz(GRID)[selected] - centers[:, None, :]) / radius
            keys_f = self.key[level](source.features)[selected]
            values = self.value[level](source.features)[selected]
            logits = (query[:, None, :] * keys_f).sum(-1) / math.sqrt(query.shape[-1])
            logits = (logits + self.relative[level](relative).squeeze(-1)).masked_fill(~valid, -1e4)
            weights = torch.softmax(logits, -1) * valid
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            parts.append((weights[..., None] * values).sum(1))
        hidden = self.hidden(torch.cat(parts, -1))
        raw = self.slots(hidden).reshape(-1, self.config.proposals_per_anchor, 4)
        xyz = pointer_xyz_from_offsets(
            centers, raw[..., :3], self.config.offset_limit_m,
            self.config.spherical_offsets)
        guide = hidden if self.config.pointer_voxel_mode else self.guide(hidden)
        return SurfaceProposals(xyz, raw[..., 3], centers, batch,
                                F.normalize(guide, dim=-1),
                                self.config.pointer_voxel_mode,
                                self.config.threshold)


@dataclass
class SurfaceStudentOutput:
    conditioning: SparseSites
    surface: SurfaceProposals
    full_confidence_field: bool = False

    def exported(self) -> dict:
        p = self.surface
        exported = {
            "conditioning_features": self.conditioning.features,
            "conditioning_coords": self.conditioning.coords,
            "surface_xyz": p.xyz,
            "surface_scores": p.score,
            "surface_logits": p.logits,
            "surface_anchor_xyz": p.anchor_xyz,
            "surface_anchor_batch": p.anchor_batch,
            "surface_radar_guide": p.guide,
            "grid_metadata": {"origin_xyz_m": GRID.minimum_xyz,
                              "voxel_xyz_m": GRID.size_xyz,
                              "conditioning_stride": 8,
                              "coordinate_order": "batch,z,y,x"},
            "interface_version": "crossmodal_surface_proposals_v1",
        }
        if p.pointer_voxel_mode:
            field = merge_surface_pointers(p)
            if self.full_confidence_field:
                exported.update({"confidence_coords": field.coords,
                                 "confidence_probabilities": field.confidence[:, None],
                                 "confidence_logits": torch.logit(field.confidence.clamp(1e-6, 1 - 1e-6))[:, None],
                                 "confidence_radar_support_count": field.source_count,
                                 "interface_version": "crossmodal_fine_association_v1"})
                return exported
            selected = field.confidence >= p.selection_threshold
            exported.update({"confidence_coords": field.coords[selected],
                             "confidence_scores": field.confidence[selected],
                             "confidence_radar_support_count": field.source_count[selected],
                             "confidence_threshold": p.selection_threshold,
                             "interface_version": "crossmodal_surface_proposals_v2_pointer_voxels"})
        return exported


class RadarOnlySurfaceStudent(nn.Module):
    def __init__(self, radar_checkpoint: str, config: SurfaceStudentConfig | None = None):
        super().__init__()
        self.config = config or SurfaceStudentConfig()
        self.radar_encoder = FrozenPreviousStage1Sparse(radar_checkpoint, modality="radar")
        self.shared = ResidualAdapter(self.config.feature_dim)
        self.feature_head = ResidualAdapter(self.config.feature_dim)
        self.feature_norm = nn.LayerNorm(self.config.feature_dim)
        self.surface_head = RadarSurfaceHead(self.config)

    def load_feature_student(self, checkpoint: dict) -> None:
        """Install and freeze the separately distilled radar-only representation."""
        if checkpoint.get("version") != "crossmodal_feature_student_v1":
            raise ValueError("Expected a trained feature-only radar student checkpoint")
        if (Path(checkpoint.get("radar_checkpoint", "")).resolve()
                != Path(self.radar_encoder.source_checkpoint).resolve()):
            raise ValueError("Surface decoder and student use different radar encoders")
        if checkpoint.get("config", {}).get("feature_dim") != self.config.feature_dim:
            raise ValueError("Student and surface feature dimensions differ")
        from .train_teacher import _hash
        if _hash(self.radar_encoder.source_checkpoint) != checkpoint.get("radar_sha256"):
            raise ValueError("Frozen radar encoder source changed since feature distillation")
        state = checkpoint["model"]
        for key, value in self.radar_encoder.state_dict().items():
            if not torch.equal(value.cpu(), state["radar_encoder." + key].cpu()):
                raise ValueError("Student checkpoint radar encoder weights differ from the source")
        for name in ("shared", "feature_head", "feature_norm"):
            prefix = name + "."
            module_state = {key[len(prefix):]: value for key, value in state.items()
                            if key.startswith(prefix)}
            getattr(self, name).load_state_dict(module_state, strict=True)
            getattr(self, name).requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.radar_encoder.eval()
        return self

    def forward(self, radar_points: torch.Tensor,
                radar_valid: torch.Tensor) -> SurfaceStudentOutput:
        with torch.no_grad():
            radar, levels = self.radar_encoder.forward_pyramid(radar_points, radar_valid)
        shared = radar.replace_features(self.shared(radar.features))
        conditioned = shared.replace_features(
            self.feature_norm(self.feature_head(shared.features)))
        surface = self.surface_head((*levels, conditioned if self.config.decoder_from_conditioned
                                     else shared))
        return SurfaceStudentOutput(conditioned, surface)
