"""The radar-only ablation must use exactly the Stage-II 12/1.6 m gate."""

from pathlib import Path

import numpy as np
import torch

from models.radar_lidar_stage2.sparse_geometry import GRID as STAGE2_GRID
from models.radar_lidar_crossmodal_v1.meta_cache import cache_manifest
from scripts.cache_meta_occupancy_fast import (
    _clean_points_in_gate, _gate_on_global_cells,
)


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
