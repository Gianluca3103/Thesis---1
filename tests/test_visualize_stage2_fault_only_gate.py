"""The active-region viewer encodes complete, distinct voxel masks."""

import base64

import numpy as np
import pytest
import torch

from models.radar_lidar_stage2.sparse_geometry import GRID
from scripts.visualize_stage2_fault_only_gate import (
    OBSERVED_BIT, candidate_payload, render_frame)


def test_viewer_packs_every_radar_voxel_and_fault_only_subset():
    raw = np.zeros(GRID.shape_zyx, dtype=np.bool_)
    active = np.zeros_like(raw)
    raw[0, 80, 50] = True
    raw[0, 80, 51] = True
    raw[0, 110, 50] = True
    active[0, 80, 50] = True
    observed = np.array([np.ravel_multi_index((0, 80, 51), GRID.shape_zyx)])
    encoded, count = candidate_payload(raw, active, observed)
    records = np.frombuffer(
        base64.b64decode(encoded),
        dtype=np.dtype([("index", "<u4"), ("membership", "<u4")]))
    assert count == 3
    assert len(records) == 3
    assert np.all(records["membership"] & 1)
    by_index = dict(zip(records["index"].tolist(),
                        records["membership"].tolist()))
    assert by_index[np.ravel_multi_index((0, 80, 50), GRID.shape_zyx)] & 2
    assert not by_index[np.ravel_multi_index((0, 110, 50), GRID.shape_zyx)] & 2
    assert by_index[int(observed[0])] & OBSERVED_BIT


def test_viewer_rejects_active_voxel_outside_radar_gate():
    raw = np.zeros(GRID.shape_zyx, dtype=np.bool_)
    active = raw.copy()
    active[0, 80, 50] = True
    with pytest.raises(ValueError, match="subset"):
        candidate_payload(raw, active, np.empty(0, dtype=np.int64))


def test_viewer_defaults_to_fault_only_and_labels_oracle(tmp_path):
    raw = np.zeros(GRID.shape_zyx, dtype=np.bool_)
    raw[0, 80, 50] = True
    page = tmp_path / "sample.html"
    record = render_frame(
        page, frame_id="00001", fault="partial_loss_30",
        region={"kind": "azimuth_sector", "fault_type": "partial_loss_30"},
        raw=raw, active=raw.copy(),
        clean=torch.tensor([[10.1, 0.1, -1.875, 1.0]]),
        faulty=torch.empty((0, 3)), radar=torch.empty((0, 7)),
        display_points=10, seed=1)
    html = page.read_text(encoding="utf-8")
    assert "box.checked=i===1" in html
    assert "known synthetic injector footprint" in html
    assert record["radar_cells"] == record["fault_only_cells"] == 1
