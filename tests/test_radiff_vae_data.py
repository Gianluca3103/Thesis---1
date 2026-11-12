"""Boundary tests for paper input fields, crop, and reconstruction metrics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.radiff_radar_data import (GlobalScale, RadarFiveFrames,
                                       cache_split, select_paper_radar)
from scripts.train_radiff_vae import FixedInput, matched_counts, metric_summary


class PaperRadarInputTests(unittest.TestCase):
    def test_crops_voxel_range_and_uses_compensated_doppler(self):
        raw = np.array([
            [1, 2, 0, 9, 99, -3, 7],
            [-0.1, 2, 0, 9, 99, -3, 7],
            [51.2, 2, 0, 9, 99, -3, 7],
            [1, 2, -3, 5, 10, 4, 0],
        ], dtype=np.float32)
        selected = select_paper_radar(raw)
        np.testing.assert_array_equal(selected, [[1, 2, 0, -3, 9],
                                                 [1, 2, -3, 4, 5]])

    def test_real_five_sweep_pilot_and_invertible_scale(self):
        vod = Path(__file__).resolve().parents[2] / "View-Of-Delft dataset" / "view_of_delft_PUBLIC"
        if not vod.is_dir():
            self.skipTest("Local VoD copy is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = cache_split(vod, root, "train", limit=2)
            second = cache_split(vod, root, "train", limit=2)
            self.assertEqual((first["written"], second["reused"]), (2, 2))
            frames = RadarFiveFrames(root, "train")
            raw = frames.load(0)
            self.assertEqual(raw.shape[1], 5)
            scale = GlobalScale.fit(frames)
            self.assertLess(float((scale.decode(scale.encode(raw)) - raw).abs().max()), 1e-4)
            sample = FixedInput(frames, scale, 1024, training=False)[0]
            self.assertEqual(tuple(sample["input"].shape), (1024, 5))
            self.assertEqual(tuple(sample["raw"].shape), tuple(raw.shape))
            self.assertGreater(len(sample["raw"]), 14)

    def test_paper_roi_voxel_iou_counts_outside_predictions_as_false_positives(self):
        target = np.array([[0.1, 0.0, -2.9, 1.0, 2.0]], dtype=np.float32)
        predicted = np.array([[0.1, 0.0, -2.9, 1.0, 2.0],
                              [-0.1, 0.0, -2.9, 3.0, 4.0]], dtype=np.float32)
        counts = matched_counts(predicted, target)
        self.assertEqual(counts["voxel_tp"], 1)
        self.assertEqual(counts["outside_predicted_points"], 1)
        self.assertEqual(counts["predicted_voxels"], 2)
        summary = metric_summary(counts)
        self.assertAlmostEqual(summary["voxel_iou"], 0.5)
        self.assertAlmostEqual(summary["paper_nearest_rcs_mae"], 1.0)
        self.assertAlmostEqual(summary["paper_nearest_compensated_doppler_mae"], 1.0)


if __name__ == "__main__":
    unittest.main()
