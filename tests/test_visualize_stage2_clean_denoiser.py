"""Compact samples must show gated occupancy and remain browser-readable."""

import numpy as np
import torch

from scripts.visualize_stage2_clean_denoiser import (
    _html, _png, select_examples)


def test_sample_selection_spans_low_middle_and_high_iou():
    rows = [{"frame_id": f"{index:05d}", "iou": index / 10}
            for index in range(10)]
    chosen = select_examples(rows, 6)
    assert len(chosen) == 6
    assert chosen[0]["iou"] == 0.0
    assert chosen[-1]["iou"] == 0.9
    assert len({row["frame_id"] for row in chosen}) == 6


def test_static_and_interactive_views_keep_gate_and_oracle_distinct(tmp_path):
    fine = torch.tensor([[0, 8, 80, 10], [0, 8, 80, 11]])
    clean = np.array([[2.1, 0.1, 0.1]], dtype=np.float32)
    radar = np.array([[2.2, 0.2, 0.0]], dtype=np.float32)
    occupied = np.array([True, False])
    predicted = np.array([False, True])
    oracle = np.array([True, False])
    png = tmp_path / "val_00001.png"
    viewer = tmp_path / "val_00001.html"
    _png(png, frame_id="00001", split="val", clean=clean, radar=radar,
         fine=fine, occupied=occupied, predicted=predicted, oracle=oracle,
         iou=0.0, oracle_iou=1.0)
    _html(viewer, frame_id="00001", split="val", epoch=125,
          threshold=0.82, iou=0.0, oracle_iou=1.0,
          fine=fine, clean=clean, radar=radar, occupied=occupied,
          predicted=predicted, oracle=oracle, seed=3103)
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    page = viewer.read_text(encoding="utf-8")
    assert "Denoiser false positives" in page
    assert "Frozen LPCR oracle" in page
    assert "No faulty LiDAR is used" in page
    assert "0.82" in page
