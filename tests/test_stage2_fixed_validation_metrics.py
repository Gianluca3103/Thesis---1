"""Fixed validation reports fault-gate and full-scene occupancy separately."""

import pytest
import torch

from scripts import evaluate_stage2_conditional_reconstruction as evaluator
from scripts.train_stage2_conditional_from_scratch import mse_epoch_log


def test_fixed_validation_reports_gate_and_full_scene(monkeypatch):
    gate = torch.tensor([[0, 0, 80, 50], [0, 0, 80, 51]])
    monkeypatch.setattr(evaluator, "predict_batch", lambda *args, **kwargs: {
        "fine": gate, "logits": torch.tensor([-2.0, 2.0])})
    clean = torch.tensor([[10.1, 0.1, -1.875],
                          [10.3, 0.1, -1.875],
                          [20.1, 0.1, -1.875]])
    faulty = clean[[0, 2]]
    monkeypatch.setattr(evaluator, "reference_clean", lambda *args: clean)
    rows = [[{"frame_id": "00001", "faulty_xyz": faulty}]]

    report = evaluator.fixed_threshold_validation(
        None, rows, None, torch.ones(2), device="cpu", seed=1, steps=2)

    assert report["fault_area"]["faulty"]["iou"] == pytest.approx(0.5)
    assert report["fault_area"]["completed"]["iou"] == pytest.approx(1.0)
    assert report["fault_area"]["completed"]["iou_0p2m"] == pytest.approx(1.0)
    assert report["faulty_baseline_iou"] == pytest.approx(2 / 3)
    assert report["iou"] == pytest.approx(1.0)
    assert report["full_iou_improvement_pp"] == pytest.approx(100 / 3)
    assert "generated_decoder_loss" not in report


def test_mse_log_includes_requested_metrics_and_skipped_epochs():
    full = {"added": 0,
            "fault_area": {"faulty": {"iou": 0.4, "iou_0p2m": 0.5},
                           "completed": {"iou": 0.6, "iou_0p2m": 0.7}},
            "faulty_baseline_iou": 0.68,
            "faulty_baseline_iou_0p2m": 0.69,
            "iou": 0.70, "iou_0p2m": 0.72,
            "full_iou_improvement_pp": 2.0,
            "full_iou_0p2m_improvement_pp": 3.0}
    message = mse_epoch_log(5, 0.2, 0.3, full, 0.7)
    for label in ("train_noise_MSE=0.2000", "val_noise_MSE=0.3000",
                  "fault_area_exact_IoU=40.00%->60.00%",
                  "fault_area_IoU@0.2m=50.00%->70.00%",
                  "full_exact_IoU=68.00%->70.00% (+2.00 pp)",
                  "full_IoU@0.2m=69.00%->72.00% (+3.00 pp)"):
        assert label in message
    assert "reconstruction_val=skipped" in mse_epoch_log(
        2, 0.2, 0.3, None, 0.7)
