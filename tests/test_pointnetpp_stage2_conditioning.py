"""Frozen radar anchors must reach Stage II without clean LiDAR inputs."""

import json
from pathlib import Path

import pytest
import torch

from models.radar_lidar_stage2.geometry_gate import geometry_gate_output
from models.radar_lidar_stage2.clean_denoising import batched_hierarchy
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import (
    LATENT_CHANNELS, SparseResidualStage2)
from scripts.cache_pointnetpp_stage2_features import RadarFeatureCache, VERSION
from scripts.train_stage2_clean_denoiser import _conditioned_inputs


def test_feature_cache_rejects_wrong_encoder_and_preserves_empty_frames(tmp_path: Path):
    root = tmp_path / "train"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "version": VERSION, "frame_ids": ["00001"],
        "radar_checkpoint_sha256": "abc", "feature_channels": 1024,
        "anchors": 512}))
    torch.save({"version": VERSION, "frame_id": "00001",
                "xyz": torch.empty((0, 3)),
                "features": torch.empty((0, 1024), dtype=torch.float16)},
               root / "00001.pt")
    cache = RadarFeatureCache(tmp_path, "train", "abc")
    xyz, features = cache.get("00001")
    assert xyz.shape == (0, 3) and features.shape == (0, 1024)
    with pytest.raises(ValueError, match="Incompatible"):
        RadarFeatureCache(tmp_path, "train", "different")


def test_learned_radar_features_reach_sparse_latent():
    mask = __import__("numpy").zeros(GRID.shape_zyx, dtype=bool)
    mask[5, 15, 15] = True
    output = geometry_gate_output(
        torch.empty((0, 7)), torch.empty((0, 3)),
        {"kind": "full_scan"}, mask, 12)
    output["conditioning_xyz"] = torch.tensor([[3.2, -12.8, -0.4]])
    output["conditioning_features"] = torch.ones((1, 1024))
    # Exercise the conditioning method without constructing CUDA MinkowskiEngine
    # components, which are unavailable in the local Windows test environment.
    model = SparseResidualStage2.__new__(SparseResidualStage2)
    torch.nn.Module.__init__(model)
    model.grid = GRID
    model.radar_feature_channels = 1024
    model.radar_adapter = torch.nn.Linear(1024, LATENT_CHANNELS)
    with torch.no_grad():
        model.radar_adapter.weight.fill_(1 / 1024)
        model.radar_adapter.bias.zero_()
    condition = model.conditions(output, 0.5)
    assert condition.radar_features.shape == (1, LATENT_CHANNELS)
    assert torch.all(condition.radar_features > 0)


def test_training_condition_uses_frozen_features_instead_of_fixed_statistics():
    mask = __import__("numpy").zeros(GRID.shape_zyx, dtype=bool)
    mask[5, 15, 15] = True
    hierarchy = batched_hierarchy([torch.tensor([[5, 15, 15]])], "cpu")
    seen = {}

    class StubModel:
        def lidar_encoder(self, points, coarse, ids):
            return torch.zeros((len(coarse), LATENT_CHANNELS)), torch.zeros(
                len(coarse), dtype=torch.bool)

        def conditions(self, output, threshold):
            seen["xyz"] = output["conditioning_xyz"]
            seen["features"] = output["conditioning_features"]
            return type("Condition", (), {
                "hierarchy": hierarchy,
                "radar_features": torch.ones((len(hierarchy[-1]), LATENT_CHANNELS))})()

    features = torch.full((1, 1024), 3.0)
    row = {"radar": torch.empty((0, 7)),
           "faulty_xyz": torch.empty((0, 3)),
           "support_mask": mask,
           "learned_radar_xyz": torch.tensor([[3.2, -12.8, -0.4]]),
           "learned_radar_features": features}
    _, _, conditioned = _conditioned_inputs(
        StubModel(), [row], hierarchy, "cpu", train_radar_adapter=True)
    assert torch.equal(seen["features"], features)
    assert torch.equal(seen["xyz"], row["learned_radar_xyz"])
    assert conditioned.shape == (1, LATENT_CHANNELS)
