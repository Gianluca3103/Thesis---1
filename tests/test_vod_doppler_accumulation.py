import unittest

import numpy as np

from Fault_Localization_Model.vod_dataset.doppler_accumulation import (
    radial_compensate_verified_stack,
)


class VoDDopplerAccumulationTests(unittest.TestCase):
    def test_radial_shift_preserves_current_and_measured_features(self):
        source = np.array([
            [10, 0, 0, 5, 6, 4, 0],
            [0, 10, 1, 6, 5, -2, 0],
            [10, 10, 2, 7, 4, 0, 0],
            [5, -5, 3, 8, 3, 1, 0],
        ], dtype=np.float32)
        aligned = source.copy()
        aligned[:, :3] += np.array([2, 3, 0], dtype=np.float32)
        aligned[:, 6] = -2
        current = np.array([[8, 9, 1, 2, 1, 3, 0]], dtype=np.float32)
        stack = np.concatenate([aligned, current])
        full, stats = radial_compensate_verified_stack(
            stack, {-2: source, 0: current}, frame_period_s=0.1,
            dispersion_tolerance_m=None,
        )
        np.testing.assert_allclose(full[0, :3], [12.8, 3, 0], atol=1e-5)
        np.testing.assert_allclose(full[1, :3], [2, 12.6, 1], atol=1e-5)
        np.testing.assert_array_equal(full[:, 3:], stack[:, 3:])
        np.testing.assert_array_equal(full[-1], current[0])
        self.assertEqual(stats["window_rejected"], 0)

    def test_window_keeps_recent_evidence_and_rejects_only_over_budget(self):
        source = np.array([
            [10, 0, 0, 5, 6, 4, 0],
            [0, 10, 1, 6, 5, -2, 0],
            [10, 10, 2, 7, 4, 0, 0],
            [5, -5, 3, 8, 3, 1, 0],
        ], dtype=np.float32)
        aligned = source.copy()
        aligned[:, 6] = -2
        current = np.array([[8, 9, 1, 2, 1, 3, 0]], dtype=np.float32)
        output, stats = radial_compensate_verified_stack(
            np.concatenate([aligned, current]), {-2: source, 0: current},
            frame_period_s=0.1, dispersion_tolerance_m=0.5,
        )
        self.assertEqual(len(output), 4)
        self.assertEqual(stats["window_rejected"], 1)
        np.testing.assert_array_equal(output[-1], current[0])

    def test_rejects_corrupted_correspondence_and_invalid_timing(self):
        source = np.array([
            [10, 0, 0, 5, 6, 4, 0],
            [0, 10, 1, 6, 5, -2, 0],
            [10, 10, 2, 7, 4, 0, 0],
        ], dtype=np.float32)
        aligned = source.copy()
        aligned[:, 6] = -1
        corrupted = aligned.copy()
        corrupted[0, 3] = 999
        with self.assertRaises(ValueError):
            radial_compensate_verified_stack(
                corrupted, {-1: source}, frame_period_s=0.1,
            )
        with self.assertRaises(ValueError):
            radial_compensate_verified_stack(
                aligned, {-1: source}, frame_period_s=0,
            )


if __name__ == "__main__":
    unittest.main()
