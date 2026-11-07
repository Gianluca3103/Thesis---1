"""Fourteen-rule geometry comparison and toggleable viewer invariants."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from models.radar_lidar_crossmodal_v1.radar_pruner import (
    radar_support_mask, unique_radar_voxel_centers,
)
from scripts.compare_radar_geometry_radii import CENTERS
from scripts.compare_radar_support_rules import (
    OBSERVED_BIT, RULES, RULE_NAMES, _viewer_html, nearest_distances, selected_mask,
    write_viewer_frame,
)


def _radar(points):
    values = torch.zeros((len(points), 7), dtype=torch.float32)
    if points:
        values[:, :3] = torch.tensor(points, dtype=torch.float32)
    return values


def test_distinct_voxels_and_nested_rules():
    assert RULE_NAMES[:5] == tuple(f"{count}@0.8m" for count in range(1, 6))
    assert len(RULES) == 14
    assert OBSERVED_BIT == 1 << len(RULES)
    radar = _radar([(10.0, 0, 0)] * 4 + [(10.6, 0, 0), (11.2, 0, 0)])
    distance = nearest_distances(radar)
    assert np.isinf(distance[:, 3:]).all()
    masks = {name: selected_mask(distance, count, radius)
             for name, (count, radius) in zip(RULE_NAMES, RULES)}
    for radius, counts in ((0.8, (1, 2, 3, 4, 5)), (1.6, (1, 2, 3, 10)),
                           (3.2, (1, 2, 3, 4, 5))):
        for small, large in zip(counts, counts[1:]):
            assert np.all(~masks[f"{large}@{radius:g}m"] |
                          masks[f"{small}@{radius:g}m"])
    for count in (1, 2):
        assert np.all(~masks[f"{count}@0.8m"] | masks[f"{count}@1.6m"])
    for count in (1, 2, 3):
        assert np.all(~masks[f"{count}@1.6m"] | masks[f"{count}@3.2m"])
    np.testing.assert_array_equal(masks["3@1.6m"],
                                  radar_support_mask(radar, 1.6).ravel())
    assert len(CENTERS) == len(distance)


def test_empty_radar_selects_nothing():
    distance = nearest_distances(_radar([]))
    assert not any(selected_mask(distance, count, radius).any()
                   for count, radius in RULES)


def test_tenth_radar_voxel_distance_matches_direct_query():
    radar = _radar([(10.05 + 0.2 * i, 0.05, 0.05) for i in range(10)])
    radar_centers = unique_radar_voxel_centers(radar)
    assert len(radar_centers) == 10
    distance = nearest_distances(radar)
    ten_mask = selected_mask(distance, 10, 1.6)
    assert ten_mask.any()
    assert np.all(~ten_mask | selected_mask(distance, 5, 1.6))
    sample = np.flatnonzero(selected_mask(distance, 5, 1.6))[:1000]
    direct, _ = cKDTree(radar_centers).query(
        CENTERS[sample], k=10, distance_upper_bound=1.6 + 1e-9)
    np.testing.assert_allclose(distance[sample, 5], direct[:, -1])
    with pytest.raises(ValueError, match="Invalid radar rule"):
        selected_mask(distance, 10, 3.2)


def test_viewer_has_all_fourteen_toggles_and_escapes_payload():
    payload = {"frame_id": "<script>", "fault": "fog_sim_2",
               "rules": [{"label": name, "raw_cells": 0, "candidates": 0,
                          "missing_recall": 0} for name in RULE_NAMES],
               "candidate_count": 0, "packed_candidates": "",
               "observed_bit": OBSERVED_BIT,
               "grid": {"shape": (16, 160, 160), "minimum": (0, -16, -2),
                        "size": (0.2, 0.2, 0.25)},
               "faulty": [], "missing": [], "radar": []}
    markup = _viewer_html(payload)
    assert "<script>" not in markup.split('id="data"')[1].split("</script>")[0]
    assert "for(let i=0;i<d.rules.length;i++)" in markup
    assert "Every blue square" in markup
    embedded = markup.split('id="data" type="application/json">')[1].split("</script>")[0]
    assert json.loads(embedded)["rules"] == payload["rules"]


def test_viewer_encodes_raw_and_addition_cells(tmp_path):
    distances = np.full((len(CENTERS), 6), np.inf, dtype=np.float64)
    distances[0, 0] = distances[1, 0] = 0.5
    path = tmp_path / "sample.html"
    observed = torch.tensor([[0, 0, 0]])
    missing = torch.tensor([[0, 0, 1]])
    write_viewer_frame(path, frame_id="00001", fault="partial_loss_30",
                       distances=distances, observed_index=np.array([0]),
                       clean=torch.cat((observed, missing)), faulty=observed,
                       missing=missing, radar=_radar([(10, 0, 0)]),
                       display_points=100, seed=1)
    markup = path.read_text(encoding="utf-8")
    data = json.loads(markup.split('id="data" type="application/json">')[1]
                      .split("</script>")[0])
    assert data["candidate_count"] == 2
    assert data["rules"][0]["candidates"] == 1
    assert data["rules"][0]["raw_cells"] == 2
    assert data["rules"][0]["missing_recall"] == 100
    packed = np.frombuffer(base64.b64decode(data["packed_candidates"]),
                           dtype=[("index", "<u4"), ("membership", "<u2")])
    assert packed["index"].tolist() == [0, 1]
    assert packed["membership"].tolist() == [
        2**0 + 2**5 + 2**9 + OBSERVED_BIT,
        2**0 + 2**5 + 2**9,
    ]
    assert not any(value & OBSERVED_BIT for value in packed["membership"][1:])


def test_viewer_never_subsamples_eligible_cells(tmp_path):
    distances = np.full((len(CENTERS), 6), np.inf, dtype=np.float64)
    distances[:15001, 0] = 0.5
    path = tmp_path / "all.html"
    empty = torch.empty((0, 3), dtype=torch.long)
    write_viewer_frame(path, frame_id="00002", fault="fog_sim_2",
                       distances=distances, observed_index=np.empty(0, dtype=np.int64),
                       clean=empty, faulty=empty, missing=empty,
                       radar=_radar([(10, 0, 0)]), display_points=100, seed=1)
    embedded = path.read_text(encoding="utf-8").split(
        'id="data" type="application/json">')[1].split("</script>")[0]
    data = json.loads(embedded)
    assert data["candidate_count"] == 15001
    assert len(base64.b64decode(data["packed_candidates"])) == 15001 * 6
