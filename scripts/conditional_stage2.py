"""Shared data and prediction path for deployable conditional Stage II."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner import supported_coordinates
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity as support_identity)
from models.radar_lidar_stage2.clean_denoising import batched_hierarchy
from models.radar_lidar_stage2.conditional_sampling import (
    sample_from_faulty, sample_from_noise)
from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.fault_only_gate import POLICY, fault_only_support
from models.radar_lidar_stage2.fault_region import FaultRegionCache, cache_identity
from models.radar_lidar_stage2.sparse_residual import (
    LPCR_REVISION, VERSION, SparseResidualStage2)
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.train_stage2_clean_denoiser import (
    CONDITIONED_ONE_STEP_REVISION, MIN_RADAR_VOXELS, RADIUS_M,
    _conditioned_inputs, _rows)


CONDITIONED_FRESH_REVISION = "stage2_conditioned_fresh_epsilon_full_noise_validation_v1"
CONDITIONED_FRESH_FAULT_ONLY_REVISION = (
    "stage2_conditioned_fresh_epsilon_fault_only_gate_v1")
CONDITIONED_FAULTY_START_REVISION = "stage2_conditioned_faulty_start_clean_target_v1"
CONDITIONED_FAULT_ONLY_REVISION = "stage2_conditioned_faulty_start_fault_only_gate_v1"


class DeploymentRows:
    """Supply radar gate, measured radar, and faulty LiDAR; no clean points."""

    def __init__(self, dataset, support: RadarSupportCache, faulty_root: Path,
                 split: str, fault_regions: FaultRegionCache | None = None):
        self.dataset = dataset
        self.support = support
        self.faulty_root = faulty_root
        self.split = split
        self.fault_regions = fault_regions

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.dataset[index]
        frame_id = item["frame_id"]
        mask = self.support.load(frame_id)
        radar_supported_cells = int(mask.sum())
        if self.fault_regions is not None:
            mask = fault_only_support(mask, self.fault_regions.get(frame_id))
        return {"frame_id": frame_id,
                "fine_zyx": supported_coordinates(mask),
                "support_mask": mask,
                "radar_supported_cells": radar_supported_cells,
                "active_cells": int(mask.sum()),
                "radar": item["radar"],
                "faulty_xyz": load_faulty(
                    self.faulty_root, self.split, frame_id)}


def load_model(lpcr_path: Path, checkpoint: Path, device: str):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = saved.get("identity")
    if (not isinstance(identity, dict) or
            identity.get("revision") not in
            (CONDITIONED_ONE_STEP_REVISION, CONDITIONED_FRESH_REVISION,
             CONDITIONED_FRESH_FAULT_ONLY_REVISION,
             CONDITIONED_FAULTY_START_REVISION,
             CONDITIONED_FAULT_ONLY_REVISION) or
            identity.get("lpcr_sha256") != sha256(lpcr_path)):
        raise ValueError("Expected a conditioned epsilon-denoiser checkpoint")
    pretrained = torch.load(lpcr_path, map_location="cpu", weights_only=False)
    if (pretrained.get("version") != VERSION or
            pretrained.get("model_revision") != LPCR_REVISION or
            pretrained.get("phase") != "lpcr" or
            not pretrained.get("lpcr_full_region")):
        raise ValueError("Expected the frozen full-region LPCR checkpoint")
    model = SparseResidualStage2(diffusion_target="epsilon").to(device)
    missing, unexpected = model.load_state_dict(pretrained["model"], strict=False)
    if unexpected or any(not key.startswith("addition_head.") for key in missing):
        raise ValueError("LPCR checkpoint differs from Stage II architecture")
    model.diffusion.load_state_dict(saved["diffusion"])
    model.radar_adapter.load_state_dict(saved["radar_adapter"])
    model.eval().requires_grad_(False)
    return model, identity, saved


def deployment_loader(vod_root: Path, raw_root: Path, support_root: Path,
                      faulty_root: Path, split: str, identity: dict,
                      num_workers: int, limit: int | None = None,
                      fault_region_root: Path | None = None):
    if identity.get("fault_region_policy") == POLICY and fault_region_root is None:
        raise ValueError("Checkpoint requires the injector fault-region cache")
    if (identity.get("fault_region_policy") not in (None, POLICY) or
            (identity.get("revision") in
             (CONDITIONED_FAULT_ONLY_REVISION,
              CONDITIONED_FRESH_FAULT_ONLY_REVISION) and
             identity.get("fault_region_policy") != POLICY)):
        raise ValueError("Unknown or incomplete fault-region gate policy")
    if split in ("train", "val"):
        for root, key in ((raw_root, "data_manifest_sha256"),
                          (support_root, "support_manifest_sha256"),
                          (faulty_root, "faulty_manifest_sha256")):
            if identity[key][split] != sha256(root / split / "manifest.json"):
                raise ValueError(f"{split} cache differs from the checkpoint")
    dataset = paired_dataset(vod_root, split, identity["radar_variant"],
                             include_clean=False, data_cache_root=raw_root)
    ids = [frame.frame_id for frame in dataset.frames]
    expected = support_identity(
        split, ids, identity["radar_variant"], data_cache_root=raw_root,
        radius_m=RADIUS_M, min_radar_voxels=MIN_RADAR_VOXELS)
    support = RadarSupportCache(support_root / split, expected)
    support.check_complete()
    regions = None
    if fault_region_root is not None:
        expected_region = cache_identity(
            split, ids, faulty_root / split / "manifest.json",
            raw_root / split / "manifest.json")
        regions = FaultRegionCache(fault_region_root / split, expected_region)
        if regions.regions is None:
            raise FileNotFoundError(f"Injector footprints not built: {regions.regions_path}")
        if identity.get("fault_region_policy") == POLICY and split in ("train", "val"):
            if (identity["fault_region_sha256"][split] !=
                    sha256(regions.regions_path)):
                raise ValueError(f"{split} injector footprints differ from checkpoint")
    manifest = json.loads((faulty_root / split / "manifest.json")
                          .read_text(encoding="utf-8"))
    if manifest.get("split") != split or manifest.get("frame_ids") != ids:
        raise ValueError(f"{split} cached faults differ from official frames")
    if split == "test":
        training = json.loads((faulty_root / "train" / "manifest.json")
                              .read_text(encoding="utf-8"))
        for key in ("version", "seed", "radar_variant", "fault_types",
                    "assignment_policy", "partial_loss_policy"):
            if manifest.get(key) != training.get(key):
                raise ValueError(f"Test mild-fault policy differs: {key}")
    rows = DeploymentRows(dataset, support, faulty_root, split, regions)
    if limit is not None:
        if limit < 1:
            raise ValueError("Frame limit must be positive")
        rows = Subset(rows, range(min(limit, len(rows))))
        ids = ids[:limit]
    loader = DataLoader(
        rows,
        batch_size=identity["batch_size"], shuffle=False,
        num_workers=num_workers, collate_fn=_rows, pin_memory=True,
        persistent_workers=num_workers > 0)
    return loader, ids, manifest


@torch.no_grad()
def predict_batch(model, rows: list[dict], alpha_bars: torch.Tensor,
                  *, device: str, seed: int, steps: int,
                  sampler: str = "gaussian_ddim",
                  timestep: int = 499) -> dict:
    """Predict from radar and faulty LiDAR without reading clean LiDAR."""
    if any("clean_xyz" in row or "clean_lidar" in row or
           "full_clean_xyz" in row for row in rows):
        raise ValueError("Prediction rows must not contain clean LiDAR")
    hierarchy = batched_hierarchy([row["fine_zyx"] for row in rows], device)
    fine, coarse = hierarchy[0], hierarchy[-1]
    if not len(coarse):
        return {"fine": fine, "logits": torch.empty(0, device=device),
                "hidden": torch.empty((0, 16), device=device)}
    conditions = _conditioned_inputs(model, rows, hierarchy, device)
    frame_ids = [row["frame_id"] for row in rows]
    if sampler == "gaussian_ddim":
        latent = sample_from_noise(model, coarse, conditions, alpha_bars,
                                   frame_ids, steps=steps, seed=seed)
    elif sampler == "faulty_latent_one_step":
        if steps != 1:
            raise ValueError("Faulty-start sampling requires one step")
        latent = sample_from_faulty(model, coarse, conditions, alpha_bars,
                                    frame_ids, timestep=timestep, seed=seed)
    else:
        raise ValueError(f"Unknown Stage II sampler: {sampler}")
    logits, _, hidden = model.lpcr(
        latent, hierarchy, latent.new_ones((len(fine), 1)),
        return_hidden=True)
    return {"fine": fine, "logits": logits, "hidden": hidden}


def reference_clean(raw_root: Path, split: str, frame_id: str) -> torch.Tensor:
    """Load clean LiDAR only after prediction, solely for metric scoring."""
    path = raw_root / split / "clean_lidar" / f"{frame_id}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("frame_id") != frame_id or saved.get("version") != 1:
        raise ValueError(f"Clean reference cache differs: {path}")
    points = saved.get("points")
    if (not isinstance(points, torch.Tensor) or points.ndim != 2 or
            points.shape[1] != 4 or not bool(torch.isfinite(points).all())):
        raise ValueError(f"Invalid clean reference points: {path}")
    return points[:, :3]
