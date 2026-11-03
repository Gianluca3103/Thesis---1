"""Hard geometry boundary and add-only contract for the radar pruner."""

from __future__ import annotations

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.radar_pruner import (
    RadarPruner, RadarPrunerConfig, VERSION_1P6M, checkpoint_payload,
    radar_support_mask, supported_coordinates)
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity)
from models.radar_lidar_stage2.sparse_geometry import select_allowed
from models.radar_lidar_stage2.stage2_io import load_stage1, stage1_frame
from scripts.train_radar_pruner_stage1 import _epoch


def _radar(xyz: list[tuple[float, float, float]]) -> torch.Tensor:
    result = torch.zeros((len(xyz), 7), dtype=torch.float32)
    if xyz:
        result[:, :3] = torch.tensor(xyz)
    return result


def test_three_distinct_voxels_and_exact_radius():
    duplicate = _radar([(10.0, 0.0, 0.0)] * 10)
    assert not radar_support_mask(duplicate).any()
    radar = _radar([(10.0, 0.0, 0.0), (10.6, 0.0, 0.0),
                    (11.2, 0.0, 0.0)])
    mask = radar_support_mask(radar)
    assert mask[8, 80, 53]
    assert not mask[8, 80, 100]
    coords = supported_coordinates(mask)
    assert len(coords) == int(mask.sum())
    assert bool(mask[coords[:, 0], coords[:, 1], coords[:, 2]].all())


def test_radar_outside_roi_can_support_boundary_voxel():
    radar = _radar([(10.0, 0.0, 4.5), (10.4, 0.0, 4.5),
                    (10.8, 0.0, 4.5)])
    assert radar_support_mask(radar)[15, 80, 52]


def test_1p6m_support_checkpoint_and_stage2_bridge(tmp_path):
    radar = _radar([(10.0, 0.0, 0.0), (10.6, 0.0, 0.0),
                    (11.2, 0.0, 0.0)])
    near = radar_support_mask(radar, 1.6)
    far = radar_support_mask(radar, 3.0)
    assert near.any() and int(near.sum()) < int(far.sum())
    assert bool((~near | far).all())
    assert identity("val", ["00001"], "radar20", radius_m=1.6) != identity(
        "val", ["00001"], "radar20", radius_m=3.0)
    model = RadarPruner(RadarPrunerConfig(support_radius_m=1.6)).eval()
    checkpoint = tmp_path / "pruner_1p6.ckpt"
    torch.save(checkpoint_payload(model, radar_variant="radar20"), checkpoint)
    loaded, saved = load_stage1(checkpoint, "cpu", "radar20")
    assert saved["version"] == VERSION_1P6M
    with torch.no_grad():
        exported = stage1_frame(loaded, radar, 1.6,
                                faulty=torch.empty((0, 3)), support_mask=near)
    allowed, _ = select_allowed(exported["confidence_coords"],
                                exported["confidence_probabilities"], 1e-6)
    coords = allowed[:, 1:].numpy()
    assert bool(near[coords[:, 0], coords[:, 1], coords[:, 2]].all())


def test_pruner_never_exports_outside_support_or_observed(tmp_path):
    radar = _radar([(10.0, 0.0, 0.0), (10.6, 0.0, 0.0),
                    (11.2, 0.0, 0.0)])
    mask = radar_support_mask(radar)
    faulty = torch.tensor([[10.5, 0.1, 0.125]], dtype=torch.float32)
    model = RadarPruner().eval()
    with torch.no_grad():
        output = model(radar, faulty, mask)
        exported = model.exported(output)
    coords = output["coords"].numpy()
    assert len(coords)
    assert bool(mask[coords[:, 0], coords[:, 1], coords[:, 2]].all())
    assert not bool(((coords == np.array([8, 80, 52])).all(axis=1)).any())
    allowed, _ = select_allowed(exported["confidence_coords"],
                                exported["confidence_probabilities"], 1e-6)
    chosen = allowed[:, 1:].numpy()
    assert bool(mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]].all())

    cache = RadarSupportCache(tmp_path, identity("val", ["00001"], "radar20"),
                              create=True)
    cache.write("00001", mask)
    np.testing.assert_array_equal(cache.load("00001"), mask)

    checkpoint = tmp_path / "pruner.ckpt"
    legacy = checkpoint_payload(model, radar_variant="radar20")
    legacy["config"].pop("support_radius_m")
    torch.save(legacy, checkpoint)
    loaded, _ = load_stage1(checkpoint, "cpu", "radar20")
    bridged = stage1_frame(loaded, radar, 3.0, faulty=faulty, support_mask=mask)
    bridge_coords, _ = select_allowed(bridged["confidence_coords"],
                                      bridged["confidence_probabilities"], 1e-6)
    local = bridge_coords[:, 1:].numpy()
    assert bool(mask[local[:, 0], local[:, 1], local[:, 2]].all())


def test_one_frame_train_and_validation_step(tmp_path):
    radar = _radar([(10.0, 0.0, 0.0), (10.6, 0.0, 0.0),
                    (11.2, 0.0, 0.0)])
    clean = torch.tensor([[10.5, 0.1, 0.125, 1.0],
                          [10.9, 0.1, 0.125, 1.0]])
    faulty_root = tmp_path / "faulty"
    (faulty_root / "val").mkdir(parents=True)
    clean[:1].numpy().tofile(faulty_root / "val" / "00001.bin")
    cache = RadarSupportCache(tmp_path / "support",
                              identity("val", ["00001"], "radar20", radius_m=1.6),
                              create=True)
    cache.write("00001", radar_support_mask(radar, 1.6))

    class OneFrame:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"frame_id": "00001", "radar": radar, "clean_lidar": clean}

    model = RadarPruner(RadarPrunerConfig(support_radius_m=1.6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    expected = {"00001": "partial_loss_50"}
    train = _epoch(model, OneFrame(), None, cache, faulty_root, "val", "cpu",
                   optimizer, workers=0, epoch=1, frame_faults=expected,
                   desired_recall=0.90)
    val = _epoch(model, OneFrame(), None, cache, faulty_root, "val", "cpu",
                 None, workers=0, epoch=1, frame_faults=expected,
                 desired_recall=0.90)
    assert train["frames"] == val["frames"] == 1
    assert train["eligible_missing"] == val["eligible_missing"] == 1
    assert train["all_missing"] == val["all_missing"] == 1
    assert val["precision_at_target_eligible_recall"]["tp"] == 1
    assert val["target_eligible_recall"] == 0.90
