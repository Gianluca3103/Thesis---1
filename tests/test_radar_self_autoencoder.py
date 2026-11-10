"""Radar self-reconstruction must use only retained measured radar returns."""

import json

import torch

from models.radar_lidar_stage1.sparse import SparseSites, encode_keys, voxelize
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.filtered_radar_cache import (
    FilteredRadarCache, filtered_frame)
from models.radar_lidar_crossmodal_v1.radar_self_autoencoder import (
    CHANNELS_PER_SLOT, SLOTS, decoded_points, reconstruction_counts,
    reconstruction_loss, target_slots)


def _row(x, y, z, rcs=1.0):
    return [x, y, z, rcs, 0.5, 0.4, 0.1]


def test_filtered_cache_reconstructs_exact_post_filter_input(tmp_path):
    raw = tmp_path / "raw"
    (raw / "train" / "radar").mkdir(parents=True)
    (raw / "train" / "manifest.json").write_text(
        json.dumps({"split": "train", "frame_ids": ["00001"]}))
    cluster = [_row(0.1 + 0.2 * (i % 4), 0.1 + 0.2 * (i // 4), 0.1)
               for i in range(12)]
    points = torch.tensor(cluster + [_row(10.1, 10.1, 0.1)])
    torch.save({"frame_id": "00001", "points": points},
               raw / "train" / "radar" / "00001.pt")
    cache = FilteredRadarCache(tmp_path / "filtered", raw, "train", create=True)
    assert cache.write(filtered_frame(raw, "train", "00001"))
    cache.check_complete()
    cached = FilteredRadarCache(tmp_path / "filtered", raw, "train")[0]
    assert cached["raw_returns"] == 13
    assert torch.equal(cached["points"], points[:12])


def test_slots_reconstruct_positions_attributes_and_multiplicity():
    points = torch.tensor([
        _row(0.11, 0.12, 0.10, 3.0),
        _row(0.15, 0.13, 0.11, 4.0),
        _row(0.41, 0.22, 0.15, 5.0),
    ], dtype=torch.float32)[None]
    valid = torch.ones((1, 3), dtype=torch.bool)
    observed, _, _, _ = voxelize(points, valid, GRID)
    extra = torch.tensor([[0, observed[0, 1], observed[0, 2], observed[0, 3] + 5]])
    coords = torch.cat((observed, extra))
    coords = coords[torch.argsort(encode_keys(coords, GRID.shape_zyx))]
    features = torch.zeros((len(coords), SLOTS * CHANNELS_PER_SLOT))
    prediction = SparseSites(coords, features, GRID.shape_zyx)
    targets = target_slots(prediction, points, valid)
    assert targets["observed_points"] == 3
    assert targets["represented_points"] == 3
    assert targets["represented_voxels"] == 2

    values = features.reshape(len(coords), SLOTS, CHANNELS_PER_SLOT)
    values[..., 0] = -10
    positive = targets["labels"].bool()
    values[..., 0][positive] = 10
    offsets = targets["continuous"][positive][:, :3]
    values[..., 1:4][positive] = torch.atanh((2 * offsets).clamp(-0.999, 0.999))
    values[..., 4:8][positive] = targets["continuous"][positive][:, 3:]
    loss, _ = reconstruction_loss(prediction, targets)
    assert float(loss) < 0.001
    decoded, occupied = decoded_points(prediction)
    assert len(decoded) == 3
    metrics = reconstruction_counts(decoded, occupied, points, valid)
    assert metrics["voxel_tp"] == 2
    assert metrics["predicted_voxels"] == 2
    assert metrics["point_tp_0p2m"] == 3
    assert metrics["chamfer_squared_sum"] < 1e-8
    assert metrics["rcs_abs_sum"] < 1e-5
    assert metrics["radial_abs_sum"] < 1e-5
    assert metrics["compensated_radial_abs_sum"] < 1e-5
    assert metrics["scan_age_abs_sum"] < 1e-5


def test_slot_ceiling_reports_unrepresentable_multi_returns():
    points = torch.tensor([_row(0.11 + i * 0.01, 0.12, 0.10) for i in range(5)])[None]
    valid = torch.ones((1, 5), dtype=torch.bool)
    coords, _, _, _ = voxelize(points, valid, GRID)
    prediction = SparseSites(coords, torch.zeros((1, SLOTS * CHANNELS_PER_SLOT)),
                             GRID.shape_zyx)
    targets = target_slots(prediction, points, valid)
    assert targets["represented_points"] == SLOTS
    assert targets["observed_points"] == 5


def test_attribute_matching_handles_unequal_point_counts():
    points = torch.tensor([_row(0.11, 0.12, 0.10, 3.0),
                           _row(2.11, 2.12, 0.10, 5.0)])[None]
    valid = torch.ones((1, 2), dtype=torch.bool)
    observed, _, _, _ = voxelize(points, valid, GRID)
    prediction = torch.tensor([[0.0, 0.11, 0.12, 0.10, 3.0, 0.5, 0.4, 0.1]])
    counts = reconstruction_counts(prediction, observed[:1], points, valid)
    assert counts["point_tp_0p2m"] == 1
    assert counts["attribute_matches"] == 1
    assert counts["rcs_abs_sum"] == 0
