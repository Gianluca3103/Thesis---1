"""The radar-only ablation must use exactly the Stage-II 12/1.6 m gate."""

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import torch

from models.radar_lidar_stage2.sparse_geometry import GRID as STAGE2_GRID
from models.radar_lidar_crossmodal_v1.meta_cache import cache_manifest
from models.radar_lidar_crossmodal_v1.radar_voxel_gate import SUPPORT_POLICY
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_stage2.candidate_domain import voxel_centers_xyz
from models.radar_lidar_crossmodal_v1.train_radar_autoencoder import (
    _checkpoint_score, collate_radar_autoencoder)
from scripts.cache_meta_occupancy_fast import (
    _clean_points_in_gate, _gate_on_global_cells,
)
from scripts.sweep_meta_occupancy_thresholds import _check_manifest


def test_stage2_gate_maps_to_global_decoder_grid():
    mask = np.zeros(STAGE2_GRID.shape_zyx, dtype=np.bool_)
    mask[0, 0, 0] = True
    mask[5, 4, 3] = True
    # Global grid starts at (-51.2,-51.2,-2); local ROI starts at (0,-16,-2).
    coords = torch.tensor([
        [0, 0, 176, 256], [0, 5, 180, 259],
        [0, 0, 176, 257], [0, 0, 175, 256], [0, 0, 336, 256],
    ])
    assert _gate_on_global_cells(coords, mask).tolist() == [True, True, False, False, False]


def test_clean_point_count_uses_same_gate_and_roi():
    mask = np.zeros(STAGE2_GRID.shape_zyx, dtype=np.bool_)
    mask[0, 0, 0] = True
    batch = {
        "clean_lidar": torch.tensor([[[0.1, -15.9, -1.9, 1.0],
                                      [0.3, -15.9, -1.9, 1.0],
                                      [-0.1, -15.9, -1.9, 1.0],
                                      [0.1, -15.9, -1.9, 1.0]]]),
        "clean_lidar_valid": torch.tensor([[True, True, True, False]]),
    }
    assert _clean_points_in_gate(batch, mask) == 1


def test_gated_manifest_records_support_and_no_lidar_weights(tmp_path: Path):
    radar = tmp_path / "radar.pth"
    radar.write_bytes(b"radar")
    manifest = cache_manifest(
        split="test", frame_ids=["00001"], vod_root=tmp_path,
        radar_variant="radar_20frames_verified_doppler_radial",
        radar_encoder=radar, lidar_encoder=None, radius_m=1.6,
        max_candidates=24, gate_support_manifest_sha256="support-hash")
    assert manifest["lidar_initial_sha256"] is None
    assert manifest["support_policy"] == "radar12vox_1p6m_stage2_roi_v1"
    assert manifest["gate_support_manifest_sha256"] == "support-hash"


def test_full_grid_manifest_has_distinct_policy(tmp_path: Path):
    radar = tmp_path / "radar.pth"
    radar.write_bytes(b"radar")
    manifest = cache_manifest(
        split="val", frame_ids=["00001"], vod_root=tmp_path,
        radar_variant="radar_20frames_verified_doppler_radial",
        radar_encoder=radar, lidar_encoder=None, radius_m=1.6,
        max_candidates=24, full_radar_gate=True)
    assert manifest["support_policy"] == SUPPORT_POLICY
    assert "gate_support_manifest_sha256" not in manifest
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    saved = {"radar_sha256": manifest["radar_initial_sha256"],
             "radar_variant": manifest["radar_variant"],
             "radius_m": 1.6, "meta_config": {"max_candidates": 24},
             "support_policy": SUPPORT_POLICY}
    dataset = SimpleNamespace(frames=[SimpleNamespace(frame_id="00001")])
    assert _check_manifest(path, saved, dataset, "val", tmp_path,
                           "radar_autoencoder") == manifest


def test_best_checkpoint_uses_reconstruction_iou_when_requested():
    validation = {"iou_full_support": 0.8, "iou_exact": 0.2}
    assert _checkpoint_score(validation, "exact") == 0.2
    assert _checkpoint_score(validation, "full_support") == 0.8


def test_prefilter_manifest_is_distinct_from_raw_radar(tmp_path: Path):
    radar = tmp_path / "radar.pth"
    radar.write_bytes(b"radar")
    common = dict(split="val", frame_ids=["00001"], vod_root=tmp_path,
                  radar_variant="radar_20frames_verified_doppler_radial",
                  radar_encoder=radar, lidar_encoder=None, radius_m=1.6,
                  max_candidates=24, full_radar_gate=True)
    raw = cache_manifest(**common)
    filtered = cache_manifest(**common, radar_input_filter=True)
    assert "radar_input_filter" not in raw
    assert filtered["radar_input_filter"] is True


def test_prefiltered_training_batch_uses_only_supported_radar_returns():
    coordinates = torch.tensor([[0, 8 + z, 256 + y, 256 + x]
                                for z in range(2) for y in range(2)
                                for x in range(3)], dtype=torch.long)
    xyz = voxel_centers_xyz(coordinates, GRID)
    isolated = xyz[:1].clone()
    isolated[:, 0] += 4.0
    radar = torch.zeros((13, 7), dtype=torch.float32)
    radar[:, :3] = torch.cat((xyz, isolated))
    geometry = {"candidate_coords": coordinates[:1],
                "observed_coords": coordinates[:1],
                "occupied": torch.ones(1, dtype=torch.bool),
                "known_free": torch.zeros(1, dtype=torch.bool),
                "support": torch.ones(1, dtype=torch.bool),
                "clean_supported_points": 1,
                "represented_clean_points_supported": 1}
    row = {"frame_id": "00001", "radar": radar, "geometry": geometry}
    unfiltered = collate_radar_autoencoder([row])
    filtered = collate_radar_autoencoder([row], radar_input_filter=True)
    assert int(unfiltered["radar_valid"].sum()) == 13
    assert filtered["radar_valid"].tolist() == [[True] * 12 + [False]]
    assert torch.equal(filtered["candidate_coords"], unfiltered["candidate_coords"])
