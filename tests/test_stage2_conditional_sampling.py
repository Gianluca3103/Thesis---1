"""Conditional sampling and full-scene threshold selection without clean input."""

import torch
from pathlib import Path

from models.radar_lidar_stage2.clean_selected_metrics import voxel_centers
from models.radar_lidar_stage2.conditional_sampling import (
    faulty_to_clean_noise_target, sample_from_faulty,
    sample_from_noise, sampling_schedule)
from scripts import conditional_stage2
from scripts import evaluate_stage2_conditional_reconstruction as evaluator
from scripts.evaluate_stage2_conditional_reconstruction import curve


def test_schedule_reaches_zero_and_sampler_is_frame_deterministic():
    schedule = sampling_schedule(8)
    assert schedule[0] == 999
    assert schedule[-1] == 0
    assert bool((schedule[:-1] > schedule[1:]).all())

    class Model:
        @staticmethod
        def diffusion(xt, faulty, radar, probability, present, timestep, coarse):
            return torch.zeros_like(xt)

    coarse = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1],
                           [1, 0, 0, 0]])
    zero = torch.zeros((3, 32))
    present = torch.zeros(3, dtype=torch.bool)
    alpha = torch.linspace(0.999, 0.01, 1000)
    joint = sample_from_noise(Model(), coarse, (zero, present, zero), alpha,
                              ["00164", "00205"], steps=8, seed=3103)
    alone = sample_from_noise(
        Model(), coarse[2:].clone().zero_(),
        (zero[2:], present[2:], zero[2:]), alpha,
        ["00205"], steps=8, seed=3103)
    assert torch.equal(joint[2:], alone)


def test_faulty_start_target_recovers_clean_and_sampler_never_uses_clean():
    faulty = torch.full((3, 32), 0.2)
    clean = torch.full((3, 32), 0.8)
    noise = torch.linspace(-1, 1, 96).reshape(3, 32)
    alpha = torch.tensor(0.5)
    xt, target = faulty_to_clean_noise_target(faulty, clean, noise, alpha)
    recovered = (xt - (1 - alpha).sqrt() * target) / alpha.sqrt()
    assert torch.allclose(recovered, clean, atol=1e-6)
    assert not torch.allclose(target, noise)

    class Model:
        @staticmethod
        def diffusion(xt, faulty, radar, probability, present, timestep, coarse):
            assert bool((timestep == 499).all())
            return (xt - alpha.sqrt() * clean[coarse[:, 0]]) / (1 - alpha).sqrt()

    coarse = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0]])
    fixed = sample_from_faulty(
        Model(), coarse, (faulty[:2], torch.ones(2, dtype=torch.bool),
                          faulty[:2]), torch.full((1000,), 0.5),
        ["00164", "00205"], timestep=499, seed=3103)
    assert torch.allclose(fixed, clean[:2], atol=1e-6)


def test_prediction_rejects_clean_points_and_accepts_faulty_radar_only(monkeypatch):
    row = {"frame_id": "00164", "fine_zyx": torch.tensor([[0, 0, 0]]),
           "faulty_xyz": torch.zeros((0, 3)), "radar": torch.zeros((0, 7)),
           "support_mask": torch.zeros(1)}
    try:
        conditional_stage2.predict_batch(None, [{**row, "clean_xyz": torch.zeros(1)}],
                                         torch.ones(1000), device="cpu", seed=1,
                                         steps=2)
        assert False, "Clean input should be rejected"
    except ValueError as error:
        assert "clean LiDAR" in str(error)

    fine = torch.tensor([[0, 0, 0, 0]])
    hierarchy = (fine,) * 5
    monkeypatch.setattr(conditional_stage2, "batched_hierarchy",
                        lambda points, device: hierarchy)
    monkeypatch.setattr(conditional_stage2, "_conditioned_inputs",
                        lambda *args: (torch.zeros((1, 32)),
                                      torch.zeros(1, dtype=torch.bool),
                                      torch.zeros((1, 32))))
    monkeypatch.setattr(conditional_stage2, "sample_from_noise",
                        lambda *args, **kwargs: torch.zeros((1, 32)))

    class Model:
        @staticmethod
        def lpcr(latent, hierarchy, probability, return_hidden):
            return torch.zeros(1), torch.zeros((1, 3)), torch.zeros((1, 16))

    result = conditional_stage2.predict_batch(
        Model(), [row], torch.ones(1000), device="cpu", seed=1, steps=2)
    assert result["logits"].shape == (1,)


def test_full_roi_curve_contains_no_addition_option():
    positive = torch.zeros(1025, dtype=torch.long)
    negative = positive.clone()
    positive[800] = 1
    negative[100] = 1
    baseline = {"tp": 1, "predicted": 1, "target": 2}
    rows = curve(positive, negative, baseline)
    assert rows[800]["iou"] == 1.0
    assert rows[-1]["threshold"] > 1
    assert rows[-1]["added"] == 0
    assert rows[-1]["iou"] == 0.5


def test_validation_predicts_before_loading_clean_reference(monkeypatch):
    events = []
    fine = torch.tensor([[0, 8, 80, 10], [0, 8, 80, 11]])
    clean = torch.from_numpy(voxel_centers(fine[:1])).float()

    def predict(model, rows, alpha, **kwargs):
        assert "clean_xyz" not in rows[0]
        events.append("prediction")
        return {"fine": fine, "logits": torch.tensor([10.0, -10.0])}

    def reference(root, split, frame_id):
        events.append("reference")
        return clean

    monkeypatch.setattr(evaluator, "predict_batch", predict)
    monkeypatch.setattr(evaluator, "reference_clean", reference)
    row = {"frame_id": "00164", "faulty_xyz": clean[:0]}
    report, role = evaluator.sweep(
        None, [[row]], ["00164"], Path("."), torch.ones(1000),
        device="cpu", seed=3103, steps=4)
    assert events == ["prediction", "reference"]
    assert report["calibration"]["iou"] == 1.0
    assert role["00164"] == "calibration"
