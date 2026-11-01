"""Fault-aware, coarse-to-fine Stage-I coordinate generator.

The network scores grid-aligned sites, expands selected parents, and returns a
fine candidate pool. Clean LiDAR can keep positive parents alive *only during
training*; it never enters feature construction or inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage2.sparse_geometry import GRID as STAGE2_GRID
from .encoder import GRID as STAGE1_GRID


VERSION = "hierarchical_stage1_gate_v1"
_OFFSETS = tuple(product((0, 1), repeat=3))


@dataclass(frozen=True)
class HierarchicalGateConfig:
    channels: int = 64
    coarse_parent_budget: int = 2400
    middle_parent_budget: int = 7000
    negative_loss_weight: float = 0.35
    tversky_weight: float = 0.5
    hard_negative_weight: float = 0.0

    def __post_init__(self) -> None:
        if (self.channels < 16 or self.channels % 8 or self.coarse_parent_budget < 1
                or self.middle_parent_budget < 1
                or self.negative_loss_weight < 0
                or self.tversky_weight < 0 or self.hard_negative_weight < 0):
            raise ValueError("Invalid hierarchical Stage-I configuration")


def shape_at(scale: int) -> tuple[int, int, int]:
    if scale not in (4, 2, 1):
        raise ValueError("Hierarchical Stage I uses scales 4, 2, and 1")
    return tuple(size // scale for size in STAGE2_GRID.shape_zyx)


def keys(coords_zyx: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    z, y, x = shape
    return (coords_zyx[:, 0].long() * y + coords_zyx[:, 1].long()) * x + coords_zyx[:, 2].long()


def decode(index: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    _, y, x = shape
    z = index // (y * x)
    rem = index % (y * x)
    return torch.stack((z, rem // x, rem % x), dim=-1)


def quantize(points_xyz: torch.Tensor, scale: int) -> torch.Tensor:
    """Return unique, sorted Stage-II ROI coordinates in z,y,x order."""
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        raise ValueError("Expected point rows with XYZ")
    shape = shape_at(scale)
    if not len(points_xyz):
        return torch.empty((0, 3), device=points_xyz.device, dtype=torch.long)
    origin = points_xyz.new_tensor(STAGE2_GRID.minimum_xyz)
    size = points_xyz.new_tensor(STAGE2_GRID.voxel_xyz) * scale
    xyz = torch.floor((points_xyz[:, :3] - origin) / size).long()
    bound = xyz.new_tensor(shape[::-1])
    valid = torch.isfinite(points_xyz[:, :3]).all(-1) & ((xyz >= 0) & (xyz < bound)).all(-1)
    rows = xyz[valid][:, [2, 1, 0]]
    return decode(torch.unique(keys(rows, shape), sorted=True), shape)


def _lookup(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Membership in sorted integer keys, preserving arbitrary query order."""
    if not len(reference):
        return torch.zeros(len(query), device=query.device, dtype=torch.bool)
    position = torch.searchsorted(reference, query)
    return ((position < len(reference)) &
            (reference[position.clamp_max(len(reference) - 1)] == query))


def target_mask(coords: torch.Tensor, target_fine: torch.Tensor, scale: int) -> torch.Tensor:
    shape = shape_at(scale)
    truth = target_fine // scale
    truth_keys = torch.unique(keys(truth, shape), sorted=True)
    return _lookup(keys(coords, shape), truth_keys)


def _point_features(points: torch.Tensor, scale: int, *, radar: bool):
    """Mean point attributes and log count in occupied cells only."""
    shape = shape_at(scale)
    width = 8 if radar else 4
    if not len(points):
        return (torch.empty(0, device=points.device, dtype=torch.long),
                points.new_empty((0, width)))
    origin = points.new_tensor(STAGE2_GRID.minimum_xyz)
    size = points.new_tensor(STAGE2_GRID.voxel_xyz) * scale
    xyz_idx = torch.floor((points[:, :3] - origin) / size).long()
    bound = xyz_idx.new_tensor(shape[::-1])
    valid = ((xyz_idx >= 0) & (xyz_idx < bound)).all(-1)
    selected = points[valid]
    rows = xyz_idx[valid][:, [2, 1, 0]]
    if not len(rows):
        return (torch.empty(0, device=points.device, dtype=torch.long),
                points.new_empty((0, width)))
    normalized = selected[:, :3] / selected.new_tensor((32.0, 16.0, 2.0))
    if radar:
        attrs = torch.stack((torch.tanh(selected[:, 3] / 30.0),
                             torch.tanh(selected[:, 4] / 15.0),
                             torch.tanh(selected[:, 5] / 15.0),
                             selected[:, 6] / 19.0), dim=-1)
        values = torch.cat((normalized, attrs), dim=-1)
    else:
        values = normalized
    unique, inverse = torch.unique(keys(rows, shape), sorted=True, return_inverse=True)
    count = values.new_zeros(len(unique)).index_add_(0, inverse, values.new_ones(len(inverse)))
    summed = values.new_zeros((len(unique), values.shape[1])).index_add_(0, inverse, values)
    features = torch.cat((summed / count[:, None], count.log1p()[:, None]), dim=-1)
    return unique, features


def _gather(query: torch.Tensor, source, shape: tuple[int, int, int], width: int) -> torch.Tensor:
    ref_keys, ref_features = source
    out = ref_features.new_zeros((len(query), width))
    if len(query) and len(ref_keys):
        query_keys = keys(query, shape)
        where = torch.searchsorted(ref_keys, query_keys)
        found = ((where < len(ref_keys)) &
                 (ref_keys[where.clamp_max(len(ref_keys) - 1)] == query_keys))
        out[found] = ref_features[where[found]]
    return out


def _choose(logits: torch.Tensor, positive: torch.Tensor | None, limit: int) -> torch.Tensor:
    """Keep high-scoring parents and, for training only, all positive parents."""
    if not len(logits):
        return torch.empty(0, device=logits.device, dtype=torch.long)
    chosen = torch.topk(logits.detach(), min(limit, len(logits))).indices
    if positive is not None:
        chosen = torch.unique(torch.cat((chosen, torch.where(positive)[0])), sorted=True)
    return chosen


class HierarchicalStage1Gate(nn.Module):
    """Generate a fine sparse candidate pool from radar and optional faulty LiDAR."""

    def __init__(self, config: HierarchicalGateConfig | None = None):
        super().__init__()
        self.config = config or HierarchicalGateConfig()
        c = self.config.channels
        self.coarse = nn.Sequential(
            nn.Conv3d(12, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv3d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv3d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU())
        self.coarse_score = nn.Conv3d(c, 1, 1)
        self.child_code = nn.Embedding(8, 16)
        self.middle = nn.Sequential(nn.Linear(c + 16 + 3 + 12, c),
                                    nn.LayerNorm(c), nn.SiLU(),
                                    nn.Linear(c, c), nn.SiLU())
        self.fine = nn.Sequential(nn.Linear(c + 16 + 3 + 12, c),
                                  nn.LayerNorm(c), nn.SiLU(),
                                  nn.Linear(c, c), nn.SiLU())
        self.middle_score = nn.Linear(c, 1)
        self.fine_score = nn.Linear(c, 1)

    def _children(self, parents: torch.Tensor, parent_features: torch.Tensor,
                  radar: torch.Tensor, faulty: torch.Tensor, scale: int,
                  layer: nn.Module, score: nn.Module):
        offsets = parents.new_tensor(_OFFSETS)
        child = (parents[:, None, :] * 2 + offsets[None]).reshape(-1, 3)
        repeated = parent_features.repeat_interleave(8, dim=0)
        child_id = torch.arange(8, device=parents.device).repeat(len(parents))
        position = (child.float() + 0.5) / child.new_tensor(shape_at(scale)).float()
        local = torch.cat((_gather(child, _point_features(radar, scale, radar=True),
                                   shape_at(scale), 8),
                           _gather(child, _point_features(faulty, scale, radar=False),
                                   shape_at(scale), 4)), dim=-1)
        hidden = layer(torch.cat((repeated, self.child_code(child_id), position, local), dim=-1))
        return child, hidden, score(hidden).squeeze(-1)

    def forward(self, radar: torch.Tensor, faulty_xyz: torch.Tensor,
                *, target_fine: torch.Tensor | None = None,
                teacher_forcing: bool = False) -> dict:
        if radar.ndim != 2 or radar.shape[1] != 7:
            raise ValueError("Radar must have seven VoD columns")
        if faulty_xyz.ndim != 2 or faulty_xyz.shape[1] != 3:
            raise ValueError("Faulty LiDAR must contain XYZ only")
        if radar.device != faulty_xyz.device:
            raise ValueError("Radar and faulty LiDAR must share a device")
        if teacher_forcing and target_fine is None:
            raise ValueError("Teacher forcing requires clean missing-site targets")
        if target_fine is not None and target_fine.device != radar.device:
            raise ValueError("Targets must share the input device")
        shape = shape_at(4)
        coarse_coords = decode(torch.arange(shape[0] * shape[1] * shape[2],
                                            device=radar.device), shape)
        raw = torch.cat((_gather(coarse_coords, _point_features(radar, 4, radar=True), shape, 8),
                         _gather(coarse_coords, _point_features(faulty_xyz, 4, radar=False),
                                 shape, 4)), dim=-1)
        x = raw.T.reshape(1, 12, *shape)
        volume = self.coarse(x)
        coarse_features = volume.flatten(2).transpose(1, 2).squeeze(0)
        coarse_logits = self.coarse_score(volume).flatten()
        coarse_positive = (target_mask(coarse_coords, target_fine, 4)
                           if teacher_forcing else None)
        chosen = _choose(coarse_logits, coarse_positive,
                         self.config.coarse_parent_budget)
        middle_coords, middle_features, middle_logits = self._children(
            coarse_coords[chosen], coarse_features[chosen], radar, faulty_xyz,
            2, self.middle, self.middle_score)
        middle_positive = (target_mask(middle_coords, target_fine, 2)
                           if teacher_forcing else None)
        chosen_mid = _choose(middle_logits, middle_positive,
                             self.config.middle_parent_budget)
        fine_coords, fine_features, fine_logits = self._children(
            middle_coords[chosen_mid], middle_features[chosen_mid], radar,
            faulty_xyz, 1, self.fine, self.fine_score)
        return {"coarse_coords": coarse_coords, "coarse_logits": coarse_logits,
                "middle_coords": middle_coords, "middle_logits": middle_logits,
                "fine_coords": fine_coords, "fine_logits": fine_logits,
                "selected_coarse_indices": chosen,
                "selected_middle_indices": chosen_mid,
                "coarse_features": coarse_features,
                "fine_features": fine_features}

    def exported(self, output: dict, *, max_sites: int,
                 threshold: float = 0.0,
                 score_mode: str = "hierarchical") -> dict:
        """Emit the existing Stage-II coordinate/feature contract for one frame."""
        if max_sites < 1 or not 0 <= threshold < 1:
            raise ValueError("Invalid final site budget or threshold")
        _, scores = fine_ranking(output, score_mode)
        chosen = torch.topk(scores, min(max_sites, len(scores))).indices
        chosen = chosen[scores[chosen] >= threshold]
        coords = output["fine_coords"][chosen]
        shift = coords.new_tensor((0, 176, 256))
        stage1_fine = coords + shift
        stage1_fine = torch.cat((stage1_fine.new_zeros((len(coords), 1)), stage1_fine), dim=1)
        coarse = output["coarse_coords"]
        stride8 = torch.stack((coarse[:, 0] // 2,
                               coarse[:, 1] // 2 + 22,
                               coarse[:, 2] // 2 + 32), dim=-1)
        encoded = keys(stride8, STAGE1_GRID.scale_shape(8))
        unique, inverse = torch.unique(encoded, sorted=True, return_inverse=True)
        features = output["coarse_features"]
        sums = features.new_zeros((len(unique), features.shape[1])).index_add_(0, inverse, features)
        counts = features.new_zeros(len(unique)).index_add_(0, inverse, features.new_ones(len(inverse)))
        conditioning = torch.cat((decode(unique, STAGE1_GRID.scale_shape(8)).new_zeros((len(unique), 1)),
                                  decode(unique, STAGE1_GRID.scale_shape(8))), dim=1)
        return {"confidence_coords": stage1_fine,
                "confidence_probabilities": scores[chosen, None],
                "conditioning_coords": conditioning,
                "conditioning_features": sums / counts[:, None],
                "grid_metadata": {"origin_xyz_m": STAGE1_GRID.minimum_xyz,
                                  "voxel_xyz_m": STAGE1_GRID.size_xyz,
                                  "conditioning_stride": 8,
                                  "coordinate_order": "batch,z,y,x"},
                "interface_version": VERSION}


def fine_ranking(output: dict, mode: str = "hierarchical") -> tuple[torch.Tensor, torch.Tensor]:
    """Rank fine children using their own and their selected ancestor evidence.

    The geometric mean of conditional probabilities is a score, not a
    calibrated probability. Thresholds still require validation calibration.
    """
    fine = output["fine_logits"]
    if mode == "fine_only":
        return fine, torch.sigmoid(fine)
    if mode != "hierarchical":
        raise ValueError("Score mode must be hierarchical or fine_only")
    middle_indices = output["selected_middle_indices"]
    coarse_indices = output["selected_coarse_indices"][middle_indices // 8]
    parent_middle = output["middle_logits"][middle_indices].repeat_interleave(8)
    parent_coarse = output["coarse_logits"][coarse_indices].repeat_interleave(8)
    if not (len(parent_middle) == len(parent_coarse) == len(fine)):
        raise ValueError("Hierarchical parent-to-child scores are misaligned")
    log_score = (F.logsigmoid(fine) + F.logsigmoid(parent_middle)
                 + F.logsigmoid(parent_coarse)) / 3
    return log_score, log_score.exp()


def stage_loss(output: dict, target_fine: torch.Tensor,
               config: HierarchicalGateConfig) -> tuple[torch.Tensor, dict]:
    terms = []
    report = {}
    for name, scale in (("coarse", 4), ("middle", 2), ("fine", 1)):
        logits = output[f"{name}_logits"]
        positive = target_mask(output[f"{name}_coords"], target_fine, scale)
        if not len(logits):
            term = output["coarse_logits"].sum() * 0
        else:
            pos = F.softplus(-logits[positive]).mean() if bool(positive.any()) else logits.sum() * 0
            neg = (F.softplus(logits[~positive]).mean() if bool((~positive).any())
                   else logits.sum() * 0)
            probability = torch.sigmoid(logits)
            tp = probability[positive].sum()
            fp = probability[~positive].sum()
            fn = positive.sum().to(probability.dtype) - tp
            tversky = 1 - (tp + 1e-6) / (tp + 0.25 * fp + 0.75 * fn + 1e-6)
            term = pos + config.negative_loss_weight * neg + config.tversky_weight * tversky
            if name == "fine" and config.hard_negative_weight:
                negative_logits = logits[~positive]
                hard = torch.topk(negative_logits,
                                  min(10000, len(negative_logits))).values
                if len(hard):
                    term = term + config.hard_negative_weight * F.softplus(hard).mean()
        terms.append(term)
        report[f"{name}_sites"] = len(logits)
        report[f"{name}_positives_reached"] = int(positive.sum())
    return sum(terms), report


def missing_target(clean_xyz: torch.Tensor, faulty_xyz: torch.Tensor) -> torch.Tensor:
    """Unique fine clean cells absent from the faulty LiDAR input."""
    clean = quantize(clean_xyz, 1)
    faulty = quantize(faulty_xyz, 1)
    present = _lookup(keys(clean, shape_at(1)), keys(faulty, shape_at(1)))
    return clean[~present]


def load_hierarchical_gate(path: str | Path, device: str = "cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("version") != VERSION:
        raise ValueError("Expected a hierarchical Stage-I checkpoint")
    model = HierarchicalStage1Gate(HierarchicalGateConfig(**saved["config"]))
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved


def checkpoint_payload(model: HierarchicalStage1Gate, **metadata) -> dict:
    return {"version": VERSION, "config": asdict(model.config),
            "model": model.state_dict(), **metadata}
