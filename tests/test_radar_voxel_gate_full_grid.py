"""The full-grid radar gate counts distinct voxels and stays within a frame."""

import torch

from models.radar_lidar_stage2.candidate_domain import voxel_centers_xyz
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.radar_pruner import radar_support_mask
from models.radar_lidar_crossmodal_v1.radar_voxel_gate import (
    DistinctRadarVoxelGate, filter_radar_valid)
from scripts.cache_meta_occupancy_fast import _gate_on_global_cells


def _radar_points():
    center = torch.tensor([0, 8, 256, 256])
    offsets = torch.tensor([
        [0, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 0], [-1, 0, 0], [1, 0, 1],
    ])
    coords = center[None].repeat(len(offsets), 1)
    coords[:, 1:] += offsets
    xyz = voxel_centers_xyz(coords, GRID)
    radar = torch.zeros((1, len(xyz), 7))
    radar[0, :, :3] = xyz
    return center, radar


def test_full_grid_gate_accepts_twelve_distinct_voxels():
    center, radar = _radar_points()
    gate = DistinctRadarVoxelGate(radar, torch.ones((1, 12), dtype=torch.bool))
    far = center.clone()
    far[3] += 20
    assert gate.contains_cells(torch.stack((center, far))).tolist() == [True, False]
    clean = torch.tensor([[[0.1, 0.1, 0.125, 1.0],
                           [4.1, 0.1, 0.125, 1.0]]])
    assert gate.count_clean_points(clean, torch.ones((1, 2), dtype=torch.bool)) == 1


def test_duplicate_radar_returns_do_not_satisfy_twelve_voxels():
    center, radar = _radar_points()
    radar[0, -1] = radar[0, 0]
    gate = DistinctRadarVoxelGate(radar, torch.ones((1, 12), dtype=torch.bool))
    assert not bool(gate.contains_cells(center[None])[0])


def test_radar_returns_from_another_frame_do_not_count():
    center, radar = _radar_points()
    radar = torch.cat((radar, radar.clone()), dim=0)
    valid = torch.ones((2, 12), dtype=torch.bool)
    valid[1] = False
    gate = DistinctRadarVoxelGate(radar, valid)
    second = center.clone()
    second[0] = 1
    assert gate.contains_cells(torch.stack((center, second))).tolist() == [True, False]


def test_radar_returns_outside_encoder_grid_do_not_count():
    center, radar = _radar_points()
    radar[0, -1, 2] = GRID.maximum_xyz[2] + 0.125
    gate = DistinctRadarVoxelGate(radar, torch.ones((1, 12), dtype=torch.bool))
    assert not bool(gate.contains_cells(center[None])[0])


def test_full_grid_gate_agrees_with_existing_stage2_rule_inside_roi():
    center, radar = _radar_points()
    far = center.clone()
    far[3] += 20
    coords = torch.stack((center, far))
    gate = DistinctRadarVoxelGate(radar, torch.ones((1, 12), dtype=torch.bool))
    stage2_mask = radar_support_mask(radar[0], 1.6, 12)
    assert torch.equal(gate.contains_cells(coords),
                       _gate_on_global_cells(coords, stage2_mask))


def test_input_filter_keeps_all_returns_in_supported_voxels_only():
    _, radar = _radar_points()
    duplicate = radar[:, :1].clone()
    isolated = radar[:, :1].clone()
    isolated[..., 0] += 4.0
    outside_grid = radar[:, :1].clone()
    outside_grid[..., 2] = GRID.maximum_xyz[2] + 0.125
    points = torch.cat((radar, duplicate, isolated, outside_grid), dim=1)
    valid = torch.ones(points.shape[:2], dtype=torch.bool)
    kept = filter_radar_valid(points, valid)
    assert kept.tolist() == [[True] * 13 + [False, False]]


def test_input_filter_uses_the_existing_full_grid_gate_definition():
    generator = torch.Generator().manual_seed(3103)
    radar = torch.zeros((1, 128, 7))
    radar[0, :, :3] = (torch.rand((128, 3), generator=generator) - 0.5) * 4
    valid = torch.ones((1, 128), dtype=torch.bool)
    gate = DistinctRadarVoxelGate(radar, valid)
    minimum = radar.new_tensor(GRID.minimum_xyz)
    size = radar.new_tensor(GRID.size_xyz)
    xyz_indices = torch.floor((radar[0, :, :3] - minimum) / size).long()
    coordinates = torch.cat((torch.zeros((128, 1), dtype=torch.long),
                             xyz_indices[:, [2, 1, 0]]), dim=1)
    assert torch.equal(filter_radar_valid(radar, valid)[0],
                       gate.contains_cells(coordinates))
