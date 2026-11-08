"""The diagnostic gate may only remove cells from radar support."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from models.radar_lidar_crossmodal_v1.radar_pruner import SHAPE
from models.radar_lidar_stage2.fault_only_gate import POLICY, fault_only_support
from scripts import conditional_stage2
from scripts import evaluate_stage2_conditional_reconstruction as evaluator
from scripts import train_stage2_clean_denoiser as train_module
from scripts.train_stage2_clean_denoiser import _CleanRows


def _mask() -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.bool_)
    mask[0, 80, 50] = True  # azimuth near zero, well inside the fault
    mask[0, 110, 50] = True  # reliable sector
    mask[0, 50, 50] = True  # reliable sector
    return mask


def test_partial_fault_retains_only_wholly_faulty_radar_cells():
    support = _mask()
    region = {"kind": "azimuth_sector", "start_rad": math.tau - 0.2,
              "end_rad": 0.2}
    result = fault_only_support(support, region)
    assert result.sum() == 1
    assert result[0, 80, 50]
    assert not result[0, 110, 50]
    assert not result[0, 50, 50]
    assert not np.any(result & ~support)
    assert np.array_equal(fault_only_support(support, {"kind": "full_scan"}),
                          support)
    assert not fault_only_support(support, {"kind": "empty"}).any()


def test_boundary_cell_touching_reliable_sector_is_removed():
    mask = np.zeros(SHAPE, dtype=np.bool_)
    mask[0, 80, 50] = True
    # Its center lies inside this sector, but its angular half diagonal crosses
    # the start boundary, so the strict gate must reject it.
    result = fault_only_support(
        mask, {"kind": "azimuth_sector", "start_rad": 0.0,
               "end_rad": 0.1})
    assert not result.any()


def test_training_and_deployment_get_the_same_filtered_coordinates(monkeypatch):
    support = _mask()
    region = {"kind": "azimuth_sector", "start_rad": math.tau - 0.2,
              "end_rad": 0.2}

    class Cache:
        def load(self, frame_id):
            assert frame_id == "00001"
            return support

        def get(self, frame_id):
            assert frame_id == "00001"
            return region

    clean = torch.tensor([[10.1, 0.1, -1.875],
                          [10.1, 6.1, -1.875]])
    dataset = [{"frame_id": "00001", "clean_lidar": torch.cat(
        (clean, torch.ones((2, 1))), 1), "radar": torch.zeros((0, 7))}]
    monkeypatch.setattr(conditional_stage2, "load_faulty",
                        lambda *args: torch.zeros((0, 3)))
    monkeypatch.setattr(train_module, "load_faulty",
                        lambda *args: torch.zeros((0, 3)))
    train = _CleanRows(dataset, Cache(), None, Path("/unused"), "val",
                       fault_regions=Cache())[0]
    deploy = conditional_stage2.DeploymentRows(
        dataset, Cache(), None, "val", Cache())[0]
    assert torch.equal(train["fine_zyx"], deploy["fine_zyx"])
    assert len(train["fine_zyx"]) == 1
    assert len(train["clean_xyz"]) == 1
    assert "clean_xyz" not in deploy
    assert np.array_equal(train["support_mask"], deploy["support_mask"])
    assert deploy["radar_supported_cells"] == 3
    assert deploy["active_cells"] == 1


def test_bad_region_and_mask_fail_closed():
    with pytest.raises(ValueError):
        fault_only_support(_mask(), {"kind": "unknown"})
    with pytest.raises(ValueError):
        fault_only_support(np.zeros((1, 1, 1), dtype=np.bool_),
                           {"kind": "full_scan"})


@pytest.mark.parametrize("revision", [
    conditional_stage2.CONDITIONED_FRESH_FAULT_ONLY_REVISION,
    conditional_stage2.CONDITIONED_FRESH_FAULT_ONLY_BCE_REVISION,
])
def test_clean_start_fault_only_checkpoint_requires_footprint_at_inference(revision):
    identity = {"revision": revision,
                "fault_region_policy": POLICY}
    with pytest.raises(ValueError, match="requires the injector"):
        conditional_stage2.deployment_loader(
            Path("/unused"), Path("/unused"), Path("/unused"),
            Path("/unused"), "val", identity, 0)


def test_full_scene_and_active_fault_metrics_are_kept_separate(monkeypatch):
    fine = torch.tensor([[0, 0, 80, 50], [0, 0, 80, 51]])
    monkeypatch.setattr(evaluator, "predict_batch",
                        lambda *args, **kwargs: {
                            "fine": fine, "logits": torch.tensor([10.0, -10.0])})
    monkeypatch.setattr(evaluator, "reference_clean",
                        lambda *args: torch.tensor([[10.1, 0.1, -1.875]]))
    row = {"frame_id": "00001", "faulty_xyz": torch.zeros((0, 3)),
           "radar_supported_cells": 3, "active_cells": 2}
    result = evaluator.selected_metrics(
        None, [[row]], Path("/unused"), "test", None,
        device="cpu", seed=1, steps=1, threshold_index=512, role=None,
        fault_names={"00001": "partial_loss_30"})
    overall = result["overall"]["test"]
    assert overall["completed"]["iou"] == 1.0
    assert overall["active_additions"]["eligible_iou"] == 1.0
    assert overall["active_additions"]["mean_active_cells"] == 2
    assert overall["active_additions"]["mean_radar_supported_cells"] == 3
