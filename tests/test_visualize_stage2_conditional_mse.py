"""Deployment sample viewer predicts before reading the clean reference."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import visualize_stage2_conditional_mse as viewer


@pytest.mark.parametrize("training_loss,bce_weight,label", [
    ("epsilon_mse_only", None, "MSE-only"),
    ("epsilon_mse_plus_generated_standard_bce", 0.1,
     "MSE + generated BCE (weight 0.1)"),
    ("epsilon_mse_plus_generated_standard_bce", 1.0,
     "MSE + generated BCE (weight 1)"),
])
def test_balanced_samples_use_only_faulty_inputs_for_prediction(
        monkeypatch, tmp_path, training_loss, bce_weight, label):
    faults = ("fog_sim_2", "partial_loss_30", "partial_loss_40",
              "partial_loss_50")
    rows = [{"frame_id": f"{index:05d}",
             "faulty_xyz": torch.empty((0, 3)),
             "radar": torch.empty((0, 3))}
            for index in range(4)]
    ids = [row["frame_id"] for row in rows]
    manifest = {"frame_faults": dict(zip(ids, faults))}
    identity = {"training_loss": training_loss,
                "decoder_bce_weight": bce_weight,
                "fault_region_policy": "synthetic_fault_gate",
                "sampler": "gaussian_ddim", "batch_size": 2,
                "seed": 3103}
    state = {"predicted": False, "records": []}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(viewer, "load_model", lambda *_: (object(), identity,
                                                          {"epoch": 30}))
    monkeypatch.setattr(viewer, "deployment_loader", lambda *_, **__: (
        SimpleNamespace(dataset=rows), ids, manifest))
    monkeypatch.setattr(viewer, "cosine_alpha_bars", lambda **_: torch.ones(1000))
    monkeypatch.setattr(viewer, "_roi_radar", lambda *_: np.empty((0, 3)))

    def predict(_model, batch, *_args, **_kwargs):
        assert all("clean_xyz" not in row and "clean_lidar" not in row
                   for row in batch)
        state["predicted"] = True
        fine = torch.tensor([[index, 8, 80, 5]
                             for index in range(len(batch))])
        return {"fine": fine, "logits": torch.full((len(batch),), 2.0)}

    def clean(_root, _split, _frame_id):
        assert state["predicted"]
        return torch.tensor([[1.01, 0.0, 0.0]])

    def candidates(fine, logits, index, _faulty, _device):
        selected = fine[:, 0] == index
        cells = fine[selected].clone()
        cells[:, 0] = 0
        return cells, logits[selected]

    def save(_root, record, _seed):
        state["records"].append(record)
        return {"frame_id": record["frame_id"], "fault": record["fault"],
                "added_cells": record["overall"]["added_cells"],
                "overall": record["overall"],
                "html": record["frame_id"] + ".html",
                "png": record["frame_id"] + ".png"}

    monkeypatch.setattr(viewer, "predict_batch", predict)
    monkeypatch.setattr(viewer, "reference_clean", clean)
    monkeypatch.setattr(viewer, "_frame_candidates", candidates)
    monkeypatch.setattr(viewer, "_save_record", save)
    monkeypatch.setattr(sys, "argv", ["viewer", "--vod-root", str(tmp_path),
                                      "--data-cache-root", str(tmp_path),
                                      "--support-cache-root", str(tmp_path),
                                      "--faulty-root", str(tmp_path),
                                      "--fault-region-root", str(tmp_path),
                                      "--lpcr-checkpoint", str(tmp_path),
                                      "--checkpoint", str(tmp_path),
                                      "--output-root", str(tmp_path / "views"),
                                      "--per-fault", "1", "--num-workers", "0"])
    viewer.main()
    report = json.loads((tmp_path / "views" / "summary.json").read_text())
    assert report["oracle_clean_latent"] is False
    assert report["training_loss"] == training_loss
    assert report["decoder_bce_weight"] == bce_weight
    assert label in (tmp_path / "views" / "index.html").read_text()
    assert len(report["samples"]) == 4
    assert {record["fault"] for record in state["records"]} == set(faults)
    assert all(record["overall"]["added_cells"] == 1
               for record in state["records"])
