"""Stage-II sparse LPCR and faulty-to-clean latent residual diffusion.

R2LDM's encoder downsampling strides are retained: (2,2,2), (2,2,2),
(2,1,1), (2,1,1). All reconstruction coordinates are selected by the
radar-only Stage-I probability map before any Stage-II decoding takes place.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_crossmodal_v1.encoder import GRID as STAGE1_GRID

from .point_distribution import points_on_allowed
from .sparse_batch import voxelize_lidar_features
from .sparse_geometry import (
    GRID, Stage2Grid, active_hierarchy, gather_features, parent_features,
    pool_allowed_probabilities, reconstruction_targets, select_allowed,
)

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None


VERSION = "stage2_sparse_residual_v1"
MODEL_REVISION = "stage2_variable_point_density_centroid_v4"
LATENT_CHANNELS = 32
POINT_MIXTURES = 8


def _require_me() -> None:
    if ME is None:
        raise RuntimeError("Stage II requires CUDA MinkowskiEngine")


def _replace(tensor, features):
    return ME.SparseTensor(features, coordinate_map_key=tensor.coordinate_map_key,
                           coordinate_manager=tensor.coordinate_manager)


def _frame_mean(values: torch.Tensor, batch_ids: torch.Tensor,
                batch_size: int) -> torch.Tensor:
    """Average per-frame site means, preserving the batch-one objective."""
    sums = values.new_zeros(batch_size).scatter_add_(0, batch_ids, values)
    counts = values.new_zeros(batch_size).scatter_add_(
        0, batch_ids, torch.ones_like(values))
    return (sums / counts.clamp_min(1)).mean()


class SparseConvNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 *, stride: tuple[int, int, int] = (1, 1, 1),
                 kernel_size: int = 3):
        super().__init__()
        _require_me()
        self.conv = ME.MinkowskiConvolution(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            dimension=3, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        y = self.conv(x)
        return _replace(y, F.silu(self.norm(y.F)))


class SparseResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        _require_me()
        self.first = SparseConvNorm(channels, channels)
        self.second = ME.MinkowskiConvolution(
            channels, channels, kernel_size=3, stride=1, dimension=3,
            bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        y = self.second(self.first(x))
        if not torch.equal(y.C, x.C):
            raise RuntimeError("Submanifold refinement changed active coordinates")
        return _replace(x, F.silu(x.F + self.norm(y.F)))


class SharedLidarEncoder(nn.Module):
    """One encoder instance is called for both clean and faulty LiDAR."""

    def __init__(self, grid: Stage2Grid = GRID):
        super().__init__()
        _require_me()
        self.grid = grid
        self.stem = SparseConvNorm(8, 16)
        self.res0 = SparseResidualBlock(16)
        self.down1 = SparseConvNorm(16, 24, stride=(2, 2, 2))
        self.res1 = SparseResidualBlock(24)
        self.down2 = SparseConvNorm(24, 32, stride=(2, 2, 2))
        self.res2 = SparseResidualBlock(32)
        self.down3 = SparseConvNorm(32, 32, stride=(2, 1, 1))
        self.res3 = SparseResidualBlock(32)
        self.down4 = SparseConvNorm(32, LATENT_CHANNELS, stride=(2, 1, 1),
                                    kernel_size=1)
        self.res4 = SparseResidualBlock(LATENT_CHANNELS)

    def forward(self, points: torch.Tensor,
                latent_coords: torch.Tensor,
                point_batch_ids: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        voxel_coords, means = voxelize_lidar_features(
            points, point_batch_ids, grid=self.grid)
        if not len(voxel_coords):
            return (points.new_zeros((len(latent_coords), LATENT_CHANNELS)),
                    torch.zeros(len(latent_coords), dtype=torch.bool,
                                device=points.device))
        x = ME.SparseTensor(means.contiguous(), coordinates=voxel_coords.int().contiguous())
        x = self.res0(self.stem(x))
        x = self.res1(self.down1(x))
        x = self.res2(self.down2(x))
        x = self.res3(self.down3(x))
        x = self.res4(self.down4(x))
        expected_stride = self.grid.downsampling_zyx[-1]
        actual_stride = tuple(int(value) for value in x.tensor_stride)
        if actual_stride != expected_stride:
            raise RuntimeError(f"LiDAR encoder stride {actual_stride} != {expected_stride}")
        compact = x.C.long().to(x.F.device).clone()
        compact[:, 1:] //= compact.new_tensor(actual_stride)
        return gather_features(compact, torch.tanh(x.F), latent_coords,
                               self.grid.shape_at(4))


class _ActiveRefine(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        _require_me()
        self.projection = nn.Linear(in_channels, out_channels)
        self.local = SparseConvNorm(out_channels, out_channels)
        self.residual = SparseResidualBlock(out_channels)

    def forward(self, coords: torch.Tensor, features: torch.Tensor,
                shape_zyx: tuple[int, int, int]) -> torch.Tensor:
        if not len(coords):
            return features.new_empty((0, self.projection.out_features))
        x = ME.SparseTensor(F.silu(self.projection(features)),
                            coordinates=coords.int().contiguous())
        x = self.residual(self.local(x))
        compact = x.C.long().to(x.F.device)
        result, found = gather_features(compact, x.F, coords, shape_zyx)
        if not bool(found.all()):
            raise RuntimeError("Sparse convolution dropped an allowed coordinate")
        return result


class AllowedSiteLPCR(nn.Module):
    """Hierarchical point reconstruction only on Stage-I-selected sites."""

    def __init__(self, grid: Stage2Grid = GRID):
        super().__init__()
        self.grid = grid
        # Levels 4 -> 3 -> 2 -> 1 -> 0 mirror the LiDAR encoder hierarchy.
        self.level4 = _ActiveRefine(LATENT_CHANNELS + 7, 32)
        self.level3 = _ActiveRefine(32 + 7, 32)
        self.level2 = _ActiveRefine(32 + 7, 32)
        self.level1 = _ActiveRefine(32 + 7, 24)
        self.level0 = _ActiveRefine(24 + 7, 16)
        self.occupancy = nn.Linear(16, 1)
        self.log_count = nn.Linear(16, 1)
        self.point_density = nn.Linear(16, POINT_MIXTURES * 7)

    @staticmethod
    def _site_features(coords: torch.Tensor, shape: tuple[int, int, int],
                       ratio: tuple[int, int, int], probability: torch.Tensor
                       ) -> torch.Tensor:
        location = ((coords[:, 1:].to(probability.dtype) + 0.5) /
                    probability.new_tensor(shape) * 2 - 1)
        phase = ((coords[:, 1:] % coords.new_tensor(ratio)).to(probability.dtype)
                 + 0.5) / probability.new_tensor(ratio) * 2 - 1
        return torch.cat((location, phase, probability), dim=1)

    def forward(self, latent_features: torch.Tensor,
                hierarchy: tuple[torch.Tensor, ...],
                fine_probabilities: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(hierarchy) != 5 or len(latent_features) != len(hierarchy[4]):
            raise ValueError("LPCR requires the five R2LDM active coordinate levels")
        if fine_probabilities.shape != (len(hierarchy[0]), 1):
            raise ValueError("LPCR needs one Stage-I probability per allowed voxel")
        pooled = [pool_allowed_probabilities(
            hierarchy[0], fine_probabilities[:, 0], hierarchy[level],
            self.grid.downsampling_zyx[level], self.grid.shape_at(level))
                  for level in range(5)]
        root = self._site_features(hierarchy[4], self.grid.shape_at(4),
                                   (1, 1, 1), pooled[4])
        hidden = self.level4(hierarchy[4], torch.cat((latent_features, root), 1),
                             self.grid.shape_at(4))
        blocks = (self.level3, self.level2, self.level1, self.level0)
        for level, block in zip((3, 2, 1, 0), blocks):
            parent_stride = self.grid.downsampling_zyx[level + 1]
            child_stride = self.grid.downsampling_zyx[level]
            ratio = tuple(a // b for a, b in zip(parent_stride, child_stride))
            hidden = parent_features(hierarchy[level + 1], hidden,
                                     hierarchy[level], self.grid.shape_at(level + 1),
                                     ratio)
            site = self._site_features(hierarchy[level], self.grid.shape_at(level),
                                        ratio, pooled[level])
            hidden = block(hierarchy[level], torch.cat((hidden, site), 1),
                           self.grid.shape_at(level))
        return self.occupancy(hidden)[:, 0], self.log_count(hidden)[:, 0], hidden

    def density_parameters(self, hidden: torch.Tensor
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.point_density(hidden).reshape(-1, POINT_MIXTURES, 7)
        return raw[:, :, 0], raw[:, :, 1:4], raw[:, :, 4:7].clamp(-3.0, 1.0)

    def expected_local_centroid(self, hidden: torch.Tensor) -> torch.Tensor:
        """Differentiable mean of the emitted point mixture in local XYZ.

        Three-point Gaussian quadrature approximates E[tanh(mu + sigma Z)]
        for each component and axis, including the component spread.
        """
        logits, means, log_scales = self.density_parameters(hidden)
        spread = math.sqrt(3.0) * log_scales.exp()
        component_centroids = (
            torch.tanh(means - spread) + 4 * torch.tanh(means)
            + torch.tanh(means + spread)) / 6
        return (F.softmax(logits, dim=1)[:, :, None]
                * component_centroids).sum(dim=1)

    def point_nll(self, hidden: torch.Tensor, site_index: torch.Tensor,
                  local_xyz: torch.Tensor, counts: torch.Tensor,
                  frame_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Permutation-invariant likelihood for every measured point in A."""
        if not len(site_index):
            return hidden.sum() * 0
        occupied = torch.where(counts > 0)[0]
        logits, means, log_scales = self.density_parameters(hidden[occupied])
        local_index = torch.searchsorted(occupied, site_index)
        target = torch.atanh(local_xyz.clamp(-0.999, 0.999))
        delta = (target[:, None, :] - means[local_index]) / log_scales[
            local_index].exp()
        component = (-0.5 * delta.square().sum(2)
                     - log_scales[local_index].sum(2)
                     - 1.5 * math.log(2 * math.pi))
        nll = -torch.logsumexp(
            F.log_softmax(logits[local_index], dim=1) + component, dim=1)
        per_site = hidden.new_zeros(len(hidden))
        per_site.index_add_(0, site_index, nll)
        return _frame_mean(per_site[occupied] / counts[occupied],
                           frame_ids[occupied], batch_size)

    @torch.no_grad()
    def emit_points(self, hidden: torch.Tensor, coords: torch.Tensor,
                    occupancy_logits: torch.Tensor, log_count: torch.Tensor,
                    occupancy_threshold: float
                    ) -> dict:
        probabilities = occupancy_logits.sigmoid()
        selected = probabilities >= occupancy_threshold
        chosen = torch.where(selected)[0]
        predicted_counts = torch.expm1(log_count[chosen].double().clamp_min(0))
        if not bool(torch.isfinite(predicted_counts).all()) or bool((
                predicted_counts >= torch.iinfo(torch.int64).max).any()):
            raise RuntimeError("Predicted point count cannot be represented as int64")
        counts = predicted_counts.round().long().clamp_min(1)
        if not len(chosen):
            return {"points_xyz": hidden.new_empty((0, 3)),
                    "coordinates": coords.new_empty((0, 4)),
                    "occupied_coordinates": coords.new_empty((0, 4)),
                    "probabilities": hidden.new_empty(0),
                    "point_counts": counts,
                    "allowed_cells": len(coords), "predicted_cells": 0,
                    "predicted_points": 0}
        logits, means, log_scales = self.density_parameters(hidden[chosen])
        local_sites = torch.repeat_interleave(
            torch.arange(len(chosen), device=hidden.device), counts)
        mixtures = torch.multinomial(
            F.softmax(logits[local_sites], dim=1), 1)[:, 0]
        centers = (hidden.new_tensor(self.grid.minimum_xyz)
                   + (coords[chosen[local_sites]][:, [3, 2, 1]].to(hidden.dtype) + 0.5)
                   * hidden.new_tensor(self.grid.voxel_xyz))
        offsets = torch.tanh(
            means[local_sites, mixtures]
            + log_scales[local_sites, mixtures].exp()
            * torch.randn((len(local_sites), 3), device=hidden.device,
                          dtype=hidden.dtype)).clamp(-0.9999, 0.9999)
        points = centers + offsets * hidden.new_tensor(self.grid.voxel_xyz) * 0.5
        return {"points_xyz": points,
                "coordinates": coords[chosen[local_sites]],
                "occupied_coordinates": coords[chosen],
                "probabilities": probabilities[chosen[local_sites]],
                "point_counts": counts,
                "allowed_cells": len(coords),
                "predicted_cells": len(chosen),
                "predicted_points": len(local_sites)}


class SparseResidualUNet(nn.Module):
    """Sparse 3D latent U-Net with a shared site set and dual diffusion heads."""

    def __init__(self, latent_shape: tuple[int, int, int]):
        super().__init__()
        _require_me()
        self.latent_shape = latent_shape
        self.timestep = nn.Embedding(1000, 32)
        self.input_projection = nn.Linear(4 * LATENT_CHANNELS + 37, 64)
        self.high = _ActiveRefine(64, 64)
        self.down = SparseConvNorm(64, 96, stride=(1, 2, 2))
        self.low = SparseResidualBlock(96)
        self.merge = _ActiveRefine(64 + 96, 64)
        self.noise_head = nn.Linear(64, LATENT_CHANNELS)
        self.residual_head = nn.Linear(64, LATENT_CHANNELS)

    def forward(self, noisy: torch.Tensor, faulty: torch.Tensor,
                radar: torch.Tensor, probability: torch.Tensor,
                faulty_present: torch.Tensor, timestep: torch.Tensor,
                coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(coords)
        if (noisy.shape != faulty.shape or noisy.shape != radar.shape
                or noisy.shape != (count, LATENT_CHANNELS)
                or probability.shape != (count, 1)
                or faulty_present.shape != (count,)):
            raise ValueError("Diffusion condition shape differs from active latent sites")
        if not count:
            return noisy, noisy
        steps = timestep.long().reshape(-1)
        if len(steps) == 1:
            time = self.timestep(steps).expand(count, -1)
        else:
            if int(coords[:, 0].max()) >= len(steps):
                raise ValueError("Missing diffusion timestep for an active batch frame")
            time = self.timestep(steps)[coords[:, 0].long()]
        location = ((coords[:, 1:].to(noisy.dtype) + 0.5) /
                    noisy.new_tensor(self.latent_shape) * 2 - 1)
        features = torch.cat((noisy, faulty, radar, probability,
                              faulty_present[:, None].to(noisy.dtype), time,
                              location), dim=1)
        # The fourth latent-width input is the learned conditioning cross term.
        cross = faulty * radar
        features = torch.cat((features, cross), dim=1)
        high = self.high(coords, F.silu(self.input_projection(features)),
                         self.latent_shape)
        x = ME.SparseTensor(high, coordinates=coords.int().contiguous())
        low = self.low(self.down(x))
        low_coords = low.C.long().to(low.F.device).clone()
        stride = tuple(int(value) for value in low.tensor_stride)
        low_coords[:, 1:] //= low_coords.new_tensor(stride)
        query = coords.long().clone()
        query[:, 1:] //= query.new_tensor((1, 2, 2))
        up, present = gather_features(low_coords, low.F, query,
                                      (self.latent_shape[0],
                                       self.latent_shape[1] // 2,
                                       self.latent_shape[2] // 2))
        if not bool(present.all()):
            raise RuntimeError("Sparse diffusion downsampling lost an active parent")
        fused = self.merge(coords, torch.cat((high, up), dim=1), self.latent_shape)
        return self.noise_head(fused), self.residual_head(fused)


@dataclass
class Stage2Conditions:
    hierarchy: tuple[torch.Tensor, ...]
    fine_probabilities: torch.Tensor
    latent_probabilities: torch.Tensor
    radar_features: torch.Tensor
    cached_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float] | None = None


class SparseResidualStage2(nn.Module):
    def __init__(self, grid: Stage2Grid = GRID):
        super().__init__()
        self.grid = grid
        self.lidar_encoder = SharedLidarEncoder(grid)
        self.lpcr = AllowedSiteLPCR(grid)
        self.radar_adapter = nn.Linear(64, LATENT_CHANNELS)
        self.diffusion = SparseResidualUNet(grid.shape_at(4))

    def conditions(self, stage1_output: dict, threshold: float) -> Stage2Conditions:
        fine, scores = select_allowed(
            stage1_output["confidence_coords"],
            stage1_output["confidence_probabilities"], threshold,
            grid=self.grid)
        levels = active_hierarchy(fine, grid=self.grid)
        coarse = levels[-1]
        pooled = pool_allowed_probabilities(
            fine, scores, coarse, self.grid.downsampling_zyx[-1],
            self.grid.shape_at(4))
        # Gather nearby Stage-I stride-8 radar sites in metric coordinates.
        xyz_index = coarse[:, [3, 2, 1]].to(scores.dtype)
        latent_size = scores.new_tensor(self.grid.voxel_xyz) * scores.new_tensor(
            self.grid.downsampling_zyx[-1][::-1])
        center = scores.new_tensor(self.grid.minimum_xyz) + (xyz_index + 0.5) * latent_size
        source_coords = stage1_output["conditioning_coords"]
        source = stage1_output["conditioning_features"]
        if source.ndim != 2 or source.shape[1] != 64 or len(source) != len(source_coords):
            raise ValueError("Stage-I radar conditioning features must be [N,64]")
        raw = source.new_zeros((len(coarse), 64))
        if len(coarse) and len(source):
            radar_centers = STAGE1_GRID.centers_xyz(source_coords, stride=8)
            distance = torch.cdist(center.float(), radar_centers.float())
            nearby_distance, indices = distance.topk(min(4, len(source)),
                                                      largest=False, dim=1)
            weights = ((nearby_distance <= 3.2).to(source.dtype) /
                       nearby_distance.to(source.dtype).clamp_min(0.1))
            raw = ((source[indices] * weights[..., None]).sum(1) /
                   weights.sum(1, keepdim=True).clamp_min(1e-8))
        return Stage2Conditions(levels, scores[:, None], pooled,
                                self.radar_adapter(raw))

    def encode_lidar(self, points: torch.Tensor,
                     conditions: Stage2Conditions,
                     point_batch_ids: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lidar_encoder(points, conditions.hierarchy[-1],
                                  point_batch_ids)

    def lpcr_loss(self, latent: torch.Tensor, conditions: Stage2Conditions,
                  clean_xyz: torch.Tensor,
                  point_weight: float = 0.04,
                  count_weight: float = 0.1,
                  centroid_weight: float = 0.04,
                  clean_batch_ids: torch.Tensor | None = None,
                  batch_size: int = 1) -> dict[str, torch.Tensor | float]:
        logits, log_count, hidden = self.lpcr(
            latent, conditions.hierarchy, conditions.fine_probabilities)
        occupied, known_free, target_offsets, coverage = (
            conditions.cached_targets if conditions.cached_targets is not None
            else reconstruction_targets(clean_xyz, conditions.hierarchy[0], grid=self.grid))
        if (occupied.shape != logits.shape or known_free.shape != logits.shape
                or target_offsets.shape != (len(logits), 3)
                or bool((occupied & known_free).any())):
            raise ValueError("LPCR labels differ from the radar-selected active cells")
        if not len(logits):
            zero = latent.sum() * 0
            return {"loss": zero, "occupancy": zero, "count": zero,
                    "centroid": zero,
                    "point_nll": zero, "count_mae": 0.0, "target_points": 0.0,
                    "iou_known": 0.0, "iou_full": 0.0,
                    "gate_coverage": coverage, "known_sites": 0,
                    "positive_sites": 0}
        site_index, local_xyz, counts = points_on_allowed(
            clean_xyz, conditions.hierarchy[0], clean_batch_ids, grid=self.grid)
        if not torch.equal(counts > 0, occupied):
            raise ValueError("Clean point multiplicity differs from cached occupancy")
        known = occupied | known_free
        frame_ids = conditions.hierarchy[0][:, 0].long()
        if batch_size == 1:
            occ = (F.binary_cross_entropy_with_logits(logits[known],
                                                      occupied[known].to(logits.dtype))
                   if bool(known.any()) else logits.sum() * 0)
            count = (F.smooth_l1_loss(log_count[occupied],
                                      torch.log1p(counts[occupied].to(log_count.dtype)))
                     if bool(occupied.any()) else log_count.sum() * 0)
        else:
            occ = _frame_mean(
                F.binary_cross_entropy_with_logits(logits[known],
                                                   occupied[known].to(logits.dtype),
                                                   reduction="none"),
                frame_ids[known], batch_size) + logits.sum() * 0
            count = _frame_mean(F.smooth_l1_loss(
                log_count[occupied], torch.log1p(counts[occupied].to(log_count.dtype)),
                reduction="none"), frame_ids[occupied], batch_size) + log_count.sum() * 0
        point_nll = self.lpcr.point_nll(
            hidden, site_index, local_xyz, counts, frame_ids, batch_size)
        if bool(occupied.any()):
            local_sums = hidden.new_zeros((len(hidden), 3))
            local_sums.index_add_(0, site_index, local_xyz.to(hidden.dtype))
            target_centroids = local_sums[occupied] / counts[occupied, None]
            predicted_centroids = self.lpcr.expected_local_centroid(hidden[occupied])
            half_voxel = hidden.new_tensor(self.grid.voxel_xyz) * 0.5
            centroid = _frame_mean(
                ((predicted_centroids - target_centroids) * half_voxel
                 ).abs().mean(1), frame_ids[occupied], batch_size)
        else:
            centroid = hidden.sum() * 0
        with torch.no_grad():
            predicted = torch.sigmoid(logits) >= 0.5
            intersection = predicted & occupied
            full_union = predicted | occupied
            known_union = (predicted & known) | occupied
            if batch_size == 1:
                iou_known = float(intersection.sum() / known_union.sum().clamp_min(1))
                iou_full = float(intersection.sum() / full_union.sum().clamp_min(1))
            else:
                sums = [torch.bincount(frame_ids[mask], minlength=batch_size)
                        for mask in (intersection, known_union, full_union)]
                iou_known = float((sums[0] / sums[1].clamp_min(1)).float().mean())
                iou_full = float((sums[0] / sums[2].clamp_min(1)).float().mean())
            predicted_count = torch.expm1(
                log_count[occupied].double().clamp_min(0)).round()
            count_mae = (float((predicted_count - counts[occupied]).abs().float().mean())
                         if bool(occupied.any()) else 0.0)
        return {"loss": (occ + count_weight * count + point_weight * point_nll
                         + centroid_weight * centroid),
                "occupancy": occ, "count": count, "point_nll": point_nll,
                "centroid": centroid,
                "count_mae": count_mae,
                "target_points": float(len(site_index)) / batch_size,
                "iou_known": iou_known, "iou_full": iou_full,
                "gate_coverage": coverage,
                "known_sites": float(known.sum()) / batch_size,
                "positive_sites": float(occupied.sum()) / batch_size}

    def diffusion_loss(self, faulty_lidar: torch.Tensor, clean_lidar: torch.Tensor,
                       conditions: Stage2Conditions, alpha_bars: torch.Tensor,
                       *, residual_weight: float = 1.0,
                       reconstruction_weight: float = 1.0,
                       generator: torch.Generator | None = None,
                       faulty_batch_ids: torch.Tensor | None = None,
                       clean_batch_ids: torch.Tensor | None = None,
                       batch_size: int = 1) -> dict:
        with torch.no_grad():
            faulty, faulty_present = self.encode_lidar(
                faulty_lidar, conditions, faulty_batch_ids)
            clean, _ = self.encode_lidar(clean_lidar, conditions, clean_batch_ids)
        residual = clean - faulty
        timestep = torch.randint(len(alpha_bars), (batch_size,), device=residual.device,
                                 generator=generator)
        latent_frame_ids = conditions.hierarchy[-1][:, 0].long()
        alpha = alpha_bars[timestep[latent_frame_ids]][:, None]
        noise = torch.randn(residual.shape, device=residual.device,
                            dtype=residual.dtype, generator=generator)
        noisy = alpha.sqrt() * residual + (1 - alpha).sqrt() * noise
        noise_hat, residual_hat = self.diffusion(
            noisy, faulty, conditions.radar_features,
            conditions.latent_probabilities, faulty_present,
            timestep, conditions.hierarchy[-1])
        noise_loss = _frame_mean((noise_hat - noise).square().mean(1),
                                 latent_frame_ids, batch_size)
        residual_loss = _frame_mean((residual_hat - residual).square().mean(1),
                                    latent_frame_ids, batch_size)
        reconstruction = self.lpcr_loss(faulty + residual_hat, conditions,
                                         clean_lidar[:, :3],
                                         clean_batch_ids=clean_batch_ids,
                                         batch_size=batch_size)
        total = (noise_loss + residual_weight * residual_loss
                 + reconstruction_weight * reconstruction["loss"])
        return {"loss": total, "noise": noise_loss,
                "residual": residual_loss, "reconstruction": reconstruction}

    @torch.no_grad()
    def sample(self, faulty_lidar: torch.Tensor, conditions: Stage2Conditions,
               alpha_bars: torch.Tensor, *, steps: int = 64,
               occupancy_threshold: float = 0.5) -> dict:
        if steps < 1 or steps > len(alpha_bars):
            raise ValueError("Sampling steps must fit the diffusion schedule")
        faulty, present = self.encode_lidar(faulty_lidar, conditions)
        residual = torch.randn_like(faulty)
        schedule = torch.linspace(len(alpha_bars) - 1, 0, steps,
                                  device=faulty.device).round().long()
        for index, timestep in enumerate(schedule):
            noise_hat, residual_hat = self.diffusion(
                residual, faulty, conditions.radar_features,
                conditions.latent_probabilities, present, timestep,
                conditions.hierarchy[-1])
            if index + 1 == len(schedule):
                residual = residual_hat
            else:
                next_alpha = alpha_bars[schedule[index + 1]]
                residual = (next_alpha.sqrt() * residual_hat
                            + (1 - next_alpha).sqrt() * noise_hat)
        logits, log_count, hidden = self.lpcr(
            faulty + residual, conditions.hierarchy, conditions.fine_probabilities)
        return self.lpcr.emit_points(
            hidden, conditions.hierarchy[0], logits, log_count,
            occupancy_threshold)


def cosine_alpha_bars(steps: int = 1000, *, device: torch.device | str = "cpu"
                      ) -> torch.Tensor:
    if steps < 2:
        raise ValueError("Diffusion schedule needs at least two steps")
    t = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float64)
    cumulative = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, 0.999)
    return torch.cumprod(1 - betas, dim=0).float()
