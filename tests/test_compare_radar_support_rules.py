"""Ten-rule geometry comparison and toggleable viewer invariants."""

from __future__ import annotations

import json

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.radar_pruner import radar_support_mask
from scripts.compare_radar_geometry_radii import CENTERS
from scripts.compare_radar_support_rules import (
    RULES, RULE_NAMES, _viewer_html, nearest_distances, selected_mask,
    write_viewer_frame,
)


def _radar(points):
    values = torch.zeros((len(points), 7), dtype=torch.float32)
    if points:
        values[:, :3] = torch.tensor(points, dtype=torch.float32)
    return values


def test_distinct_voxels_and_nested_rules():
    radar = _radar([(10.0, 0, 0)] * 4 + [(10.6, 0, 0), (11.2, 0, 0)])
    distance = nearest_distances(radar)
    assert np.isinf(distance[:, 3:]).all()
    masks = {name: selected_mask(distance, count, radius)
             for name, (count, radius) in zip(RULE_NAMES, RULES)}
    for radius, counts in ((0.8, (1, 2)), (1.6, (1, 2, 3)),
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


def test_viewer_has_all_ten_toggles_and_escapes_payload():
    payload = {"frame_id": "<script>", "fault": "fog_sim_2",
               "rules": [{"label": name, "candidates": 0, "missing_recall": 0}
                         for name in RULE_NAMES], "candidates": [],
               "membership": [], "faulty": [], "missing": [], "radar": []}
    markup = _viewer_html(payload)
    assert "<script>" not in markup.split('id="data"')[1].split("</script>")[0]
    assert "for(let i=0;i<d.rules.length;i++)" in markup
    embedded = markup.split('id="data" type="application/json">')[1].split("</script>")[0]
    assert json.loads(embedded)["rules"] == payload["rules"]


def test_viewer_omits_observed_cells_but_keeps_missing_candidates(tmp_path):
    distances = np.full((len(CENTERS), 5), np.inf, dtype=np.float64)
    distances[0, 0] = distances[1, 0] = 0.5
    path = tmp_path / "sample.html"
    observed = torch.tensor([[0, 0, 0]])
    missing = torch.tensor([[0, 0, 1]])
    write_viewer_frame(path, frame_id="00001", fault="partial_loss_30",
                       distances=distances, observed_index=np.array([0]),
                       clean=torch.cat((observed, missing)), faulty=observed,
                       missing=missing, radar=_radar([(10, 0, 0)]),
                       display_cells=100, display_points=100, seed=1)
    markup = path.read_text(encoding="utf-8")
    data = json.loads(markup.split('id="data" type="application/json">')[1]
                      .split("</script>")[0])
    assert len(data["candidates"]) == 1
    assert data["rules"][0]["candidates"] == 1
    assert data["rules"][0]["missing_recall"] == 100
    assert data["membership"] == [2**0 + 2**2 + 2**5]
