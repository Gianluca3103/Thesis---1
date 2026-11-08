"""Reliable-region losses and full-ROI threshold accounting."""

import torch
from torch import nn

from models.radar_lidar_stage2.clean_selected_metrics import voxel_centers
from models.radar_lidar_stage2.clean_thresholds import score_histograms
from models.radar_lidar_stage2.reliable_addition import (
    addition_losses, addition_masks)
from scripts.evaluate_stage2_reliable_addition import full_roi_curve
from scripts import train_stage2_reliable_addition as trainer


def test_empty_loss_protects_observed_and_missing_clean_cells():
    fine = torch.tensor([[0, 8, 80, 10], [0, 8, 80, 11],
                         [0, 8, 80, 12]])
    clean = torch.from_numpy(voxel_centers(fine[:2])).float()
    faulty = clean[:1]
    clean_ids = torch.zeros(len(clean), dtype=torch.long)
    faulty_ids = torch.zeros(len(faulty), dtype=torch.long)
    # A missing clean cell remains positive; the other unobserved cell is empty.
    masks = addition_masks(clean, clean_ids, faulty, faulty_ids, fine)
    assert masks["context"].observed.tolist() == [True, False, False]
    assert masks["positive"].tolist() == [False, True, False]
    assert masks["negative"].tolist() == [False, False, True]
    logits = torch.zeros(3, requires_grad=True)
    losses = addition_losses(logits, fine, masks, 1)
    losses["total"].backward()
    assert logits.grad[0] == 0
    assert logits.grad[1] < 0
    assert logits.grad[2] > 0


def test_full_roi_sweep_counts_only_new_cells():
    positive, negative = score_histograms(
        torch.tensor([0.8, 0.7]), torch.tensor([True, False]))
    baseline = {"predicted": 1, "target": 2, "tp": 1}
    rows = full_roi_curve(positive, negative, baseline)
    assert rows[800]["added"] == 1
    assert rows[800]["predicted"] == 2
    assert rows[800]["tp"] == 2
    assert rows[800]["iou"] == 1.0
    assert rows[700]["added"] == 2
    assert rows[700]["iou"] == 2 / 3
    assert rows[-1]["threshold"] > 1
    assert rows[-1]["added"] == 0
    assert rows[-1]["iou"] == 0.5


def test_validation_scores_completed_full_roi(monkeypatch):
    fine = torch.tensor([[0, 8, 80, 10], [0, 8, 80, 11],
                         [0, 8, 80, 12]])
    clean = torch.from_numpy(voxel_centers(fine[:2])).float()
    faulty = clean[:1]
    masks = addition_masks(
        clean, torch.zeros(2, dtype=torch.long), faulty,
        torch.zeros(1, dtype=torch.long), fine)
    monkeypatch.setattr(trainer, "decoded_addition_batch", lambda *args, **kwargs: {
        "fine": fine, "logits": torch.tensor([0.0, 5.0, -5.0]),
        "masks": masks})

    class Model:
        addition_head = nn.Linear(1, 1)

        def eval(self):
            return self

    row = {"full_clean_xyz": clean, "faulty_xyz": faulty}
    result = trainer.run_epoch(
        Model(), [[row]], None, torch.linspace(0.9, 0.1, 1000),
        device="cpu", epoch=1, seed=3103, eval_timestep=499,
        weights=(1, 1, 1))
    assert result["baseline"]["iou"] == 0.5
    assert result["completed_at_0p5"]["iou"] == 1.0
