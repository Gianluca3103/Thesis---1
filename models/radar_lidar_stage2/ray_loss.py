"""Optional, scan-level ray losses for Stage-II *additions*.

Clean returns define training rays; they are never model inputs at inference.
VoD does not provide firing IDs here, so XYZ direction matching is a
conservative proxy. Unmatched candidate sites receive no ray supervision.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F

from models.radar_lidar_stage1.sparse import encode_keys

from .sparse_batch import voxelize_means_batched
from .sparse_geometry import GRID, Stage2Grid


def _query_workers() -> int:
    """CPU threads for fixed ray association; it does not alter labels."""
    workers = int(os.environ.get("STAGE2_RAY_WORKERS", "1"))
    if not 1 <= workers <= 32:
        raise ValueError("STAGE2_RAY_WORKERS must be between 1 and 32")
    return workers


def _first_ranges(query: np.ndarray, returns: np.ndarray, *,
                  angular_chord: float, lateral_m: float) -> np.ndarray:
    """First compatible measured return, without treating unmatched rays as free."""
    first = np.full(len(query), np.inf, dtype=np.float64)
    if not len(query) or not len(returns):
        return first
    return_range = np.linalg.norm(returns, axis=1)
    good = return_range > 1e-3
    if not np.any(good):
        return first
    returns, return_range = returns[good], return_range[good]
    query_range = np.linalg.norm(query, axis=1)
    valid = np.flatnonzero(query_range > 1e-3)
    if not len(valid):
        return first
    tree = cKDTree(returns / return_range[:, None])
    workers = _query_workers()
    for start in range(0, len(valid), 8192 * workers):
        ids = valid[start:start + 8192 * workers]
        chord, index = tree.query(query[ids] / query_range[ids, None],
                                  k=min(16, len(returns)), workers=workers)
        chord = np.asarray(chord).reshape(len(ids), -1)
        index = np.asarray(index).reshape(len(ids), -1)
        compatible = ((chord <= angular_chord) &
                      (query_range[ids, None] * chord <= lateral_m))
        first[ids] = np.where(compatible, return_range[index], np.inf).min(1)
        # If the nearest 16 directions are all compatible, a still earlier
        # return may lie beyond the query limit. Skip that ambiguous ray.
        if len(returns) > 16:
            first[ids[compatible[:, -1]]] = np.nan
    return first


def addition_ray_losses(
        fine: torch.Tensor, addition_logits: torch.Tensor,
        observed: torch.Tensor, clean_occupied: torch.Tensor,
        clean_xyz: torch.Tensor, *, clean_batch_ids: torch.Tensor | None = None,
        angular_tolerance_deg: float = 1.5,
        lateral_tolerance_m: float = 0.15,
        range_margin_m: float | None = None,
        hit_enabled: bool = True, extra_enabled: bool = True,
        grid: Stage2Grid = GRID,
        collect_labels: bool = False,
        cached_labels: list[dict] | None = None,
        ) -> dict[str, torch.Tensor | int]:
    """First-hit likelihood and extra-return suppression on radar-allowed sites.

    Each missing clean return inside the gate defines a target ray. Its loss is
    -log(p_hit) - sum(log(1-p_before)) for unobserved candidates on that ray.
    Already retained points are fixed returns, not trainable predictions.
    Separately, empty candidate sites clearly behind a compatible clean first
    return incur -log(1-p). This concerns scan returns, not scene emptiness.
    The two terms are averaged per frame so mixed fault batches are balanced.
    """
    if angular_tolerance_deg <= 0 or lateral_tolerance_m <= 0:
        raise ValueError("Ray matching tolerances must be positive")
    if range_margin_m is None:
        range_margin_m = 0.5 * float(np.linalg.norm(grid.voxel_xyz))
    if range_margin_m < 0:
        raise ValueError("Ray range margin must be nonnegative")
    if fine.ndim != 2 or fine.shape[1] != 4 or addition_logits.shape != (len(fine),):
        raise ValueError("Ray losses require [N,4] cells and [N] addition logits")
    if observed.shape != (len(fine),) or clean_occupied.shape != (len(fine),):
        raise ValueError("Ray masks must match allowed cells")
    if clean_batch_ids is None:
        clean_batch_ids = torch.zeros(len(clean_xyz), dtype=torch.long,
                                      device=clean_xyz.device)
    if clean_batch_ids.shape != (len(clean_xyz),):
        raise ValueError("Clean point batch IDs differ from clean points")
    zero = addition_logits.sum() * 0
    if cached_labels is not None:
        if collect_labels:
            raise ValueError("Cannot collect and reuse ray labels together")
        frame_count = len(cached_labels)
        counts = [int(label["candidate_count"]) for label in cached_labels]
        if (not frame_count or sum(counts) != len(fine)
                or len(fine) > 1 and bool((fine[1:, 0] < fine[:-1, 0]).any())
                or torch.bincount(fine[:, 0].long(), minlength=frame_count).tolist()
                != counts):
            raise ValueError("Cached ray labels differ from batched fine cells")
        hit_terms, extra_terms = [], []
        hit_rays = extra_sites = offset = 0
        for label in cached_labels:
            count = int(label["candidate_count"])
            ids = {}
            for name in ("hit", "front", "extra"):
                values = np.asarray(label[name])
                if (values.ndim != 1 or values.dtype.kind not in "iu"
                        or bool(np.any(values < 0))
                        or bool(np.any(values >= count))):
                    raise ValueError(f"Invalid cached ray {name} indices")
                ids[name] = torch.as_tensor(values.astype(np.int64),
                                            device=addition_logits.device) + offset
            if hit_enabled and len(ids["hit"]):
                hit_rays += len(ids["hit"])
                frame_loss = F.softplus(-addition_logits[ids["hit"]]).sum()
                if len(ids["front"]):
                    frame_loss = frame_loss + F.softplus(
                        addition_logits[ids["front"]]).sum()
                hit_terms.append(frame_loss / len(ids["hit"]))
            if extra_enabled and len(ids["extra"]):
                extra_sites += len(ids["extra"])
                extra_terms.append(F.softplus(
                    addition_logits[ids["extra"]]).mean())
            offset += count
        return {"hit": torch.stack(hit_terms).sum() / frame_count
                if hit_terms else zero,
                "extra": torch.stack(extra_terms).sum() / frame_count
                if extra_terms else zero,
                "hit_rays": hit_rays, "extra_sites": extra_sites}
    if not (hit_enabled or extra_enabled) or not len(fine):
        return {"hit": zero, "extra": zero, "hit_rays": 0,
                "extra_sites": 0}

    # Geometry and ray association are fixed labels. Gradients remain entirely
    # on the addition probabilities, through the frozen LPCR to diffusion.
    with torch.no_grad():
        centers = (fine[:, [3, 2, 1]].to(torch.float64) + 0.5)
        centers = (centers * centers.new_tensor(grid.voxel_xyz)
                   + centers.new_tensor(grid.minimum_xyz))
        centers_np = centers.cpu().numpy()
        frames_np = fine[:, 0].cpu().numpy()
        observed_np = observed.cpu().numpy()
        occupied_np = clean_occupied.cpu().numpy()
        clean_np = clean_xyz.detach().cpu().numpy().astype(np.float64)
        clean_frames_np = clean_batch_ids.cpu().numpy()
        if hit_enabled:
            clean_coords, clean_means = voxelize_means_batched(
                clean_xyz, clean_batch_ids, grid=grid)
            clean_keys = encode_keys(clean_coords, grid.shape_zyx)
            allowed_keys = encode_keys(fine, grid.shape_zyx)
            positions = torch.searchsorted(clean_keys, allowed_keys)
            matched = ((positions < len(clean_keys)) &
                       (clean_keys[positions.clamp(max=max(len(clean_keys) - 1, 0))]
                        == allowed_keys)) if len(clean_keys) else torch.zeros_like(positions, dtype=torch.bool)
            hit_indices = torch.nonzero(
                clean_occupied & ~observed & matched, as_tuple=True)[0]
            hit_positions = positions[hit_indices]
            hits_np = hit_indices.cpu().numpy()
            hit_xyz = clean_means[hit_positions].detach().cpu().numpy().astype(np.float64)
        else:
            hits_np = np.empty(0, dtype=np.int64)
            hit_xyz = np.empty((0, 3), dtype=np.float64)

    chord_limit = 2 * np.sin(np.deg2rad(angular_tolerance_deg) / 2)
    frame_count = max(int(fine[:, 0].max()) + 1, 1)
    hit_terms = []
    extra_terms = []
    hit_rays = 0
    extra_sites = 0
    labels = ([{"candidate_count": int(np.count_nonzero(frames_np == frame)),
                "hit": np.empty(0, dtype=np.int32),
                "front": np.empty(0, dtype=np.int32),
                "extra": np.empty(0, dtype=np.int32)}
               for frame in range(frame_count)] if collect_labels else None)
    for frame in range(frame_count):
        frame_ids = np.flatnonzero(frames_np == frame)
        clean_returns = clean_np[clean_frames_np == frame]
        if not len(frame_ids) or not len(clean_returns):
            continue
        free_ids = frame_ids[~observed_np[frame_ids]]
        frame_hit_ids = hits_np[frames_np[hits_np] == frame]
        frame_hit_xyz = hit_xyz[frames_np[hits_np] == frame]

        if hit_enabled and len(frame_hit_ids):
            # A target whose ray already has a nearer clean return is an
            # ambiguous multi-return/beam-proxy match; do not force a hit.
            first_hit = _first_ranges(frame_hit_xyz, clean_returns,
                                      angular_chord=chord_limit,
                                      lateral_m=lateral_tolerance_m)
            hit_range = np.linalg.norm(frame_hit_xyz, axis=1)
            valid_hit = (np.isfinite(first_hit) &
                         (first_hit + range_margin_m >= hit_range))
            frame_hit_ids = frame_hit_ids[valid_hit]
            frame_hit_xyz = frame_hit_xyz[valid_hit]
            if len(frame_hit_ids):
                hit_rays += len(frame_hit_ids)
                # Never turn a measured clean cell into a negative merely
                # because its voxel center resembles a neighboring beam.
                front_candidate_ids = free_ids[~occupied_np[free_ids]]
                candidate_xyz = centers_np[front_candidate_ids]
                candidate_range = np.linalg.norm(candidate_xyz, axis=1)
                valid_candidate = candidate_range > 1e-3
                directions = candidate_xyz[valid_candidate] / candidate_range[valid_candidate, None]
                candidate_ids = front_candidate_ids[valid_candidate]
                candidate_range = candidate_range[valid_candidate]
                front_ids = []
                if len(candidate_ids):
                    tree = cKDTree(directions)
                    for target in frame_hit_xyz:
                        distance = np.linalg.norm(target)
                        neighbors = tree.query_ball_point(target / distance,
                                                          chord_limit)
                        if not neighbors:
                            continue
                        neighbors = np.asarray(neighbors, dtype=np.int64)
                        near = ((candidate_range[neighbors] * np.linalg.norm(
                            directions[neighbors] - target / distance, axis=1)
                            <= lateral_tolerance_m) &
                            (candidate_range[neighbors] + range_margin_m < distance))
                        front_ids.extend(candidate_ids[neighbors[near]].tolist())
                frame_loss = F.softplus(-addition_logits[
                    torch.as_tensor(frame_hit_ids, device=addition_logits.device)]).sum()
                if front_ids:
                    frame_loss = frame_loss + F.softplus(addition_logits[
                        torch.as_tensor(front_ids, device=addition_logits.device)]).sum()
                hit_terms.append(frame_loss / len(frame_hit_ids))
                if labels is not None:
                    labels[frame]["hit"] = np.searchsorted(
                        frame_ids, frame_hit_ids).astype(np.int32)
                    labels[frame]["front"] = np.searchsorted(
                        frame_ids, np.asarray(front_ids, dtype=np.int64)).astype(np.int32)

        if extra_enabled and len(free_ids):
            first = _first_ranges(centers_np[free_ids], clean_returns,
                                  angular_chord=chord_limit,
                                  lateral_m=lateral_tolerance_m)
            behind = ((np.linalg.norm(centers_np[free_ids], axis=1)
                       > first + range_margin_m) & ~occupied_np[free_ids])
            selected = free_ids[behind]
            if len(selected):
                extra_sites += len(selected)
                extra_terms.append(F.softplus(addition_logits[
                    torch.as_tensor(selected, device=addition_logits.device)]).mean())
                if labels is not None:
                    labels[frame]["extra"] = np.searchsorted(
                        frame_ids, selected).astype(np.int32)
    hit = torch.stack(hit_terms).sum() / frame_count if hit_terms else zero
    extra = torch.stack(extra_terms).sum() / frame_count if extra_terms else zero
    result = {"hit": hit, "extra": extra, "hit_rays": hit_rays,
              "extra_sites": extra_sites}
    if labels is not None:
        result["labels"] = labels
    return result
