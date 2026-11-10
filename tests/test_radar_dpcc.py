"""Focused contract checks for the radar-only D-PCC adaptation."""

import torch
from torch.utils.data import DataLoader

from models.radar_lidar_crossmodal_v1.radar_dpcc import (
    RadarDPCC, build_hierarchy, reconstruction_loss)
from models.radar_lidar_crossmodal_v1.train_radar_dpcc import (
    collate_rows, epoch)


def _frame(count=32):
    points = torch.zeros((count, 7), dtype=torch.float32)
    index = torch.arange(count, dtype=torch.float32)
    points[:, 0] = index % 8 * 0.22
    points[:, 1] = torch.div(index, 8, rounding_mode="floor") * 0.24
    points[:, 2] = 0.1
    points[:, 3] = 4.0 + index / 10
    points[:, 4] = 0.5
    points[:, 5] = 0.4
    points[:, 6] = 0.2
    return {"frame_id": "synthetic", "points": points,
            "hierarchy": build_hierarchy(points)}


def test_hierarchy_covers_every_point_and_has_three_levels():
    row = _frame()
    counts = [len(stage["sample"]) for stage in row["hierarchy"]]
    assert counts == [16, 8, 4]
    for stage, n in zip(row["hierarchy"], [32, 16, 8]):
        assert int(stage["counts"].sum()) == n
        assert bool((stage["counts"] >= 1).all())
        assert bool(stage["mask"].any(1).all())


def test_identical_xyz_still_yields_distinct_nonempty_anchors():
    points = torch.zeros((32, 7))
    hierarchy = build_hierarchy(points)
    for stage, n in zip(hierarchy, (32, 16, 8)):
        assert int(stage["counts"].sum()) == n
        assert bool((stage["counts"] >= 1).all())
        assert bool(torch.isfinite(stage["mean_distance"]).all())


def test_forward_backward_and_same_radar_metrics():
    row = _frame()
    model = RadarDPCC(width=16, max_children=8)
    result = model(row["points"], row["hierarchy"])
    loss, terms = reconstruction_loss(result, row["points"], row["hierarchy"])
    assert result.latent_count == 4
    assert result.attributes.shape == (len(result.xyz), 4)
    assert len(result.xyz) > 0
    assert torch.isfinite(loss)
    assert set(terms) == {"chamfer", "features", "intermediate", "density"}
    loss.backward()
    assert model.embed[0].weight.grad is not None
    assert bool(torch.isfinite(model.embed[0].weight.grad).all())


def test_epoch_uses_existing_radar_point_and_voxel_metrics():
    row = _frame()
    loader = DataLoader([row], batch_size=1, collate_fn=collate_rows)
    model = RadarDPCC(width=16)
    with torch.no_grad():
        report = epoch(model, loader, None, "cpu", threshold=0.5,
                       name="synthetic D-PCC val")
    assert report["frames"] == 1
    assert 0 <= report["voxel_iou"] <= 1
    assert 0 <= report["point_f1_0p2m"] <= 1
    assert report["target_points"] == 32
    assert report["mean_latent_points"] == 4
