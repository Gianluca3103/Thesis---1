"""Ray supervision uses only compatible measured rays and addition logits."""

import torch

from models.radar_lidar_stage2.ray_loss import addition_ray_losses
from models.radar_lidar_stage2.sparse_geometry import GRID


def _centers(coords: torch.Tensor) -> torch.Tensor:
    return ((coords[:, [3, 2, 1]].float() + 0.5)
            * torch.tensor(GRID.voxel_xyz) + torch.tensor(GRID.minimum_xyz))


def test_hit_front_and_extra_behind_have_separate_gradients():
    # All four sites lie close to one clean return ray. The last is off-ray.
    fine = torch.tensor([[0, 8, 80, 20], [0, 8, 80, 25],
                         [0, 8, 80, 30], [0, 8, 85, 30]])
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
    clean = _centers(fine[1:2])
    result = addition_ray_losses(
        fine, logits, torch.zeros(4, dtype=torch.bool),
        torch.tensor([False, True, False, False]), clean)
    assert result["hit_rays"] == 1
    assert result["extra_sites"] == 1
    result["hit"].backward(retain_graph=True)
    assert logits.grad[0] > 0  # free before the clean return
    assert logits.grad[1] < 0  # missing clean return is encouraged
    assert logits.grad[2] == 0  # first-hit loss stops at the hit
    assert logits.grad[3] == 0  # unrelated ray remains unknown
    logits.grad.zero_()
    result["extra"].backward()
    assert logits.grad[0] == 0
    assert logits.grad[1] == 0
    assert logits.grad[2] > 0  # added return behind first return
    assert logits.grad[3] == 0


def test_retained_hit_is_fixed_and_unknown_beams_are_not_negative():
    fine = torch.tensor([[0, 8, 80, 25], [0, 8, 80, 30],
                         [0, 8, 85, 30]])
    logits = torch.zeros(3, requires_grad=True)
    result = addition_ray_losses(
        fine, logits, torch.tensor([True, False, False]),
        torch.tensor([True, False, False]), _centers(fine[:1]))
    assert result["hit_rays"] == 0
    assert result["hit"] == 0
    assert result["extra_sites"] == 1
    result["extra"].backward()
    assert logits.grad.tolist() == [0.0, 0.5, 0.0]


def test_no_returns_create_no_free_space_or_extra_returns():
    fine = torch.tensor([[0, 8, 80, 25]])
    logits = torch.zeros(1, requires_grad=True)
    result = addition_ray_losses(
        fine, logits, torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool), torch.empty((0, 3)))
    assert result["hit_rays"] == 0
    assert result["extra_sites"] == 0
    assert result["hit"] == 0
    assert result["extra"] == 0


def test_zero_weight_path_can_skip_all_ray_geometry():
    fine = torch.tensor([[0, 8, 80, 25]])
    logits = torch.ones(1, requires_grad=True)
    result = addition_ray_losses(
        fine, logits, torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool), torch.empty((0, 3)),
        hit_enabled=False, extra_enabled=False)
    assert result["hit"] == 0
    assert result["extra"] == 0


def test_ray_matching_never_crosses_batch_frames():
    fine = torch.tensor([[0, 8, 80, 25], [1, 8, 80, 30]])
    logits = torch.zeros(2, requires_grad=True)
    clean = _centers(fine[:1])
    result = addition_ray_losses(
        fine, logits, torch.zeros(2, dtype=torch.bool),
        torch.tensor([True, False]), clean,
        clean_batch_ids=torch.tensor([0]))
    assert result["hit_rays"] == 1
    assert result["extra_sites"] == 0
    result["hit"].backward()
    assert logits.grad[1] == 0
