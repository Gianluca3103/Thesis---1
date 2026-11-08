import pytest

from scripts.compare_stage2_input_sources import compare


def _report(sampler, iou):
    baseline = {"frames": 2, "exact_tp": 3, "predicted": 4,
                "eligible_target": 5, "iou": 0.5, "f1": 0.6,
                "iou_0p2m": 0.5, "f1_0p2m": 0.6}
    completed = {**baseline, "iou": iou, "f1": iou + 0.1,
                 "iou_0p2m": iou + 0.2, "f1_0p2m": iou + 0.3}
    row = {"faulty": baseline, "completed": completed}
    return {"sampler": sampler, "checkpoint_sha256": "abc",
            "sweep": {"threshold": 0.5},
            "validation": {"overall": {"all_validation": row}},
            "test": {"overall": {"test": row}}}


def test_compare_keeps_both_reports_and_metric_deltas():
    result = compare(_report("gaussian_ddim", 0.4),
                     _report("faulty_latent_one_step", 0.45))
    assert result["comparison"]["validation"]["faulty_start_minus_gaussian"]["iou"] == pytest.approx(0.05)


def test_compare_rejects_different_faulty_baseline():
    old = _report("gaussian_ddim", 0.4)
    new = _report("faulty_latent_one_step", 0.45)
    new["test"]["overall"]["test"]["faulty"] = {
        **new["test"]["overall"]["test"]["faulty"], "frames": 3}
    try:
        compare(old, new)
    except ValueError as error:
        assert "frame set" in str(error)
    else:
        assert False, "Mismatched test frames must be rejected"
