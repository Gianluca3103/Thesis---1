"""Small CUDA/MinkowskiEngine Stage-II forward, backward and gate smoke test."""

from __future__ import annotations

import argparse

import torch

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
    args = parser.parse_args()
    torch.manual_seed(3103)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("MinkowskiEngine smoke test requires CUDA")
    clean = torch.rand((320, 3), device=device)
    clean = clean * clean.new_tensor((3.0, 3.0, 1.5)) + clean.new_tensor(
        (8.0, -1.5, -0.75))
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
    lpcr_grad = model.lpcr.occupancy.weight.grad
    if lpcr_grad is None or not torch.isfinite(lpcr_grad).all():
        raise AssertionError("LPCR occupancy gradient is missing or invalid")
    model.zero_grad(set_to_none=True)
    model.lidar_encoder.requires_grad_(False)
    model.lpcr.requires_grad_(False)
    second = model.diffusion_loss(faulty, clean, conditions,
                                  cosine_alpha_bars(16, device=device))
    second["loss"].backward()
    diffusion_grad = model.diffusion.noise_head.weight.grad
    adapter_grad = model.radar_adapter.weight.grad
    for name, grad in (("diffusion", diffusion_grad), ("radar adapter", adapter_grad)):
        if grad is None or not torch.isfinite(grad).all() or not bool((grad != 0).any()):
            raise AssertionError(f"{name} gradient is missing, zero, or invalid")
    sampled = model.sample(faulty, conditions, cosine_alpha_bars(16, device=device),
                           steps=4)
    if sampled["predicted_cells"] > sampled["allowed_cells"]:
        raise AssertionError("Reconstruction escaped the Stage-I gate")
    model.zero_grad(set_to_none=True)
    twin_conditions = [model.conditions(stage1, 0.3) for _ in range(2)]
    for item in twin_conditions:
        item.cached_targets = reconstruction_targets(clean, item.hierarchy[0])
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
            model.diffusion.noise_head.weight.grad is None or
            not torch.isfinite(model.diffusion.noise_head.weight.grad).all()):
        raise AssertionError("Batched diffusion loss or gradient is invalid")
    print({"lpcr_loss": float(first["loss"]),
           "diffusion_loss": float(second["loss"]),
           "batched_diffusion_loss": float(batched["loss"]),
           "allowed": sampled["allowed_cells"],
           "predicted": sampled["predicted_cells"],
           "gradients_finite": True})


if __name__ == "__main__":
    main()
