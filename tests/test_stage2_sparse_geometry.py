import torch

from models.radar_lidar_stage2.sparse_geometry import (
    GRID, active_hierarchy, gather_features, parent_features, select_allowed,
)


def test_gate_maps_only_stage1_sites_into_overlap():
    source = torch.tensor([[0, 0, 176, 256], [0, 15, 335, 415],
                           [0, 0, 175, 256], [0, 0, 176, 257]])
    probability = torch.tensor([[0.8], [0.9], [0.9], [0.1]])
    selected, scores = select_allowed(source, probability, 0.3)
    assert selected.tolist() == [[0, 0, 0, 0], [0, 15, 159, 159]]
    assert torch.allclose(scores, torch.tensor([0.8, 0.9]))
    levels = active_hierarchy(selected)
    assert [len(level) for level in levels] == [2, 2, 2, 2, 2]
    assert GRID.shape_at(4) == (1, 40, 40)


def test_coordinate_joins_preserve_gradients_and_missing_sites():
    source = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 2]])
    features = torch.tensor([[1.0], [3.0]], requires_grad=True)
    query = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]])
    found_features, present = gather_features(source, features, query, (1, 1, 3))
    assert present.tolist() == [True, False, True]
    assert found_features[:, 0].tolist() == [1.0, 0.0, 3.0]
    found_features.sum().backward()
    assert features.grad[:, 0].tolist() == [1.0, 1.0]
    parent = torch.tensor([[0, 0, 0, 0]])
    child = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]])
    repeated = parent_features(parent, torch.tensor([[5.0]]), child,
                               (1, 1, 1), (1, 1, 2))
    assert repeated[:, 0].tolist() == [5.0, 5.0]
