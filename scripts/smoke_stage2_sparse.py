"""Small CUDA/MinkowskiEngine Stage-II forward, backward and gate smoke test."""

from __future__ import annotations

import argparse

import torch

from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.full_lpcr import full_region_training_conditions
from models.radar_lidar_stage2.lpcr_eval import point_cloud_metrics
from models.radar_lidar_stage2.sparse_batch import voxelize_means_batched
from models.radar_lidar_stage2.sparse_geometry import (
    GRID, reconstruction_targets, voxelize_means,
)
from models.radar_lidar_stage2.sparse_residual import (
    SparseResidualStage2, cosine_alpha_bars,
)
from scripts.train_stage2_sparse import _combine_conditions, _combine_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-ray-loss", action="store_true",
                        help="Exercise optional ray losses; skipped by default")
    args = parser.parse_args()
    torch.manual_seed(3103)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("MinkowskiEngine smoke test requires CUDA")
    xyz = torch.rand((320, 3), device=device)
    xyz = xyz * xyz.new_tensor((3.0, 3.0, 1.5)) + xyz.new_tensor(
        (8.0, -1.5, -0.75))
    xyz[:3] = xyz.new_tensor([[8.01, -1.49, -0.74],
                              [8.04, -1.47, -0.72],
                              [8.08, -1.43, -0.69]])
    clean = xyz
    faulty = clean[::3]
    occupied, _ = voxelize_means(clean)
    neighbors = [occupied]
    for axis in (1, 2, 3):
        shifted = occupied.clone()
        shifted[:, axis] += 1
        inside = shifted[:, axis] < GRID.shape_zyx[axis - 1]
        neighbors.append(shifted[inside])
    local = torch.unique(torch.cat(neighbors), dim=0, sorted=True)
    global_coords = local.clone()
    global_coords[:, 1:] += global_coords.new_tensor((0, 176, 256))
    stage1 = {
        "confidence_coords": global_coords,
        "confidence_probabilities": torch.full((len(local), 1), 0.8,
                                                device=device),
        "conditioning_coords": torch.tensor(
            [[0, z, y, x] for z in (0, 1) for y in (31, 32)
             for x in (37, 38)], device=device),
        "conditioning_features": torch.randn((8, 64), device=device),
    }
    model = SparseResidualStage2().to(device)
    conditions = model.conditions(stage1, 0.3)
    latent, _ = model.encode_lidar(clean, conditions)
    first = model.lpcr_loss(latent, conditions, clean)
    first["loss"].backward()
    for name, layer in (("occupancy", model.lpcr.occupancy),
                        ("offset", model.lpcr.offset)):
        grad = layer.weight.grad
        if grad is None or not torch.isfinite(grad).all() or not bool((grad != 0).any()):
            raise AssertionError(f"LPCR {name} gradient is missing or invalid")
    model.zero_grad(set_to_none=True)
    full = full_region_training_conditions(clean, "smoke", 1, 8192)
    if len(full.hierarchy[0]) < 4096 or not bool(
            (full.cached_targets[0] | full.cached_targets[1]).any()):
        raise AssertionError("Full-region LPCR patch is empty")
    full_latent, _ = model.encode_lidar(clean, full)
    full_loss = model.lpcr_loss(full_latent, full, clean)
    full_loss["loss"].backward()
    if not torch.isfinite(full_loss["loss"]):
        raise AssertionError("Full-region LPCR loss is nonfinite")
    identical = point_cloud_metrics(clean[:3].cpu().numpy(),
                                    clean[:3].cpu().numpy())
    if (identical["chamfer_squared_l2"] != 0.0
            or abs(identical["fscore_5cm"] - 1.0) > 1e-6):
        raise AssertionError("R2LDM-style geometric metrics failed identity")
    model.zero_grad(set_to_none=True)
    model.lidar_encoder.requires_grad_(False)
    model.lpcr.requires_grad_(False)
    second = model.diffusion_loss(faulty, clean, conditions,
                                  cosine_alpha_bars(16, device=device))
    second["loss"].backward()
    diffusion_grad = model.diffusion.residual_head.weight.grad
    adapter_grad = model.radar_adapter.weight.grad
    addition_grad = model.addition_head.network[0].weight.grad
    for name, grad in (("diffusion", diffusion_grad), ("radar adapter", adapter_grad),
                       ("addition head", addition_grad)):
        if grad is None or not torch.isfinite(grad).all() or not bool((grad != 0).any()):
            raise AssertionError(f"{name} gradient is missing, zero, or invalid")
    if args.check_ray_loss:
        model.zero_grad(set_to_none=True)
        # The preceding backward freed the radar-adapter graph in conditions.
        ray_conditions = model.conditions(stage1, 0.3)
        ray_trial = model.diffusion_loss(
            faulty, clean, ray_conditions, cosine_alpha_bars(16, device=device),
            ray_hit_weight=0.1, ray_extra_weight=0.1)
        ray_trial["loss"].backward()
        if (not torch.isfinite(ray_trial["rays"]["hit"])
                or not torch.isfinite(ray_trial["rays"]["extra"])
                or model.addition_head.network[0].weight.grad is None
                or not torch.isfinite(model.addition_head.network[0].weight.grad).all()):
            raise AssertionError("Optional ray loss or addition gradient is invalid")
    sampled = model.sample(faulty, conditions, cosine_alpha_bars(16, device=device),
                           steps=4)
    if sampled["predicted_cells"] > sampled["allowed_cells"]:
        raise AssertionError("Reconstruction escaped the Stage-I gate")
    if (sampled["predicted_points"] != len(sampled["points_xyz"])
            or sampled["predicted_points"] != int(sampled["point_counts"].sum())
            or sampled["predicted_points"] != sampled["predicted_cells"]):
        raise AssertionError("LPCR must emit one centroid per selected voxel")
    torch.manual_seed(3103)
    raw = model.sample(faulty, conditions, cosine_alpha_bars(16, device=device),
                       steps=4, occupancy_threshold=0.01, missing_only=False)
    torch.manual_seed(3103)
    additions = model.sample(faulty, conditions,
                             cosine_alpha_bars(16, device=device),
                             steps=4, occupancy_threshold=0.01)
    if (additions["predicted_cells"] > additions["allowed_cells"] or
            raw["predicted_cells"] > raw["allowed_cells"]):
        raise AssertionError("Prediction escaped the Stage-I gate")
    faulty_voxels, _ = voxelize_means(faulty)
    if bool(torch.isin(encode_keys(additions["coordinates"], GRID.shape_zyx),
                       encode_keys(faulty_voxels, GRID.shape_zyx)).any()):
        raise AssertionError("Additions overlap an observed faulty voxel")
    baseline = model.sample(faulty, conditions,
                            cosine_alpha_bars(16, device=device),
                            steps=4, occupancy_threshold=0.01,
                            use_residual=False)
    if (baseline["residual_abs_present_latent"] not in (None, 0.0)
            or baseline["residual_abs_absent_latent"] not in (None, 0.0)):
        raise AssertionError("Zero-residual ablation did not disable diffusion")
    if len(sampled["points_xyz"]):
        minimum = sampled["points_xyz"].new_tensor(GRID.minimum_xyz)
        size = sampled["points_xyz"].new_tensor(GRID.voxel_xyz)
        ijk = torch.floor((sampled["points_xyz"] - minimum) / size).long()
        if not torch.equal(ijk[:, [2, 1, 0]], sampled["coordinates"][:, 1:]):
            raise AssertionError("A generated point escaped its allowed voxel")
    total_loss = model.diffusion_loss(
        clean[:0], clean, conditions, cosine_alpha_bars(16, device=device))
    if not torch.isfinite(total_loss["loss"]):
        raise AssertionError("Total-loss LiDAR caused a nonfinite diffusion loss")
    total_loss_sample = model.sample(
        clean[:0], conditions, cosine_alpha_bars(16, device=device), steps=4)
    if total_loss_sample["predicted_cells"] > total_loss_sample["allowed_cells"]:
        raise AssertionError("Total-loss reconstruction escaped the Stage-I gate")
    model.zero_grad(set_to_none=True)
    twin_conditions = [model.conditions(stage1, 0.3) for _ in range(2)]
    for item in twin_conditions:
        item.cached_targets = reconstruction_targets(clean[:, :3], item.hierarchy[0])
    joined = _combine_conditions(twin_conditions)
    batched_clean, clean_ids = _combine_points([clean, clean])
    batched_faulty, faulty_ids = _combine_points([faulty, faulty])
    single_voxels, _ = voxelize_means(clean)
    batch_voxels, _ = voxelize_means_batched(batched_clean, clean_ids)
    if (len(batch_voxels) != 2 * len(single_voxels) or
            set(batch_voxels[:, 0].tolist()) != {0, 1}):
        raise AssertionError("Batched LiDAR voxels merged across frames")
    batched = model.diffusion_loss(
        batched_faulty, batched_clean, joined,
        cosine_alpha_bars(16, device=device),
        faulty_batch_ids=faulty_ids, clean_batch_ids=clean_ids, batch_size=2)
    batched["loss"].backward()
    if (not torch.isfinite(batched["loss"]) or
            model.diffusion.residual_head.weight.grad is None or
            not torch.isfinite(model.diffusion.residual_head.weight.grad).all()):
        raise AssertionError("Batched diffusion loss or gradient is invalid")
    print({"lpcr_loss": float(first["loss"]),
           "full_region_lpcr_loss": float(full_loss["loss"]),
           "diffusion_loss": float(second["loss"]),
           "addition_loss": float(second["addition"]["final"]),
           "healthy_loss": float(second["addition"]["healthy"]),
           "missing_sites": second["addition"]["missing_sites"],
           "batched_diffusion_loss": float(batched["loss"]),
           "total_loss_diffusion_loss": float(total_loss["loss"]),
           "allowed": sampled["allowed_cells"],
           "predicted": sampled["predicted_cells"],
           "overlap_observed_suppressed": additions["overlap_observed_cells"],
           "gradients_finite": True})


if __name__ == "__main__":
    main()
