import unittest
import importlib.util
from pathlib import Path

import numpy as np

module_path = Path(__file__).resolve().parents[1] / "models/radar_lidar_stage2/detector_export.py"
spec = importlib.util.spec_from_file_location("detector_export", module_path)
detector_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector_export)
detector_clouds = detector_export.detector_clouds
radar_sector_indices = detector_export.radar_sector_indices
retained_clean_mask = detector_export.retained_clean_mask


class DetectorExportTests(unittest.TestCase):
    def test_retains_duplicate_multiplicity_and_original_intensity(self):
        clean = np.array([
            [1, 0, 0, 0.2], [1, 0, 0, 0.8], [2, 0, 0, 0.4],
            [-1, 0, 0, 0.7],
        ], dtype=np.float32)
        faulty, completed = detector_clouds(
            clean, np.array([0, 1, 2]),
            np.array([[1, 0, 0]], dtype=np.float32),
            np.array([[1.5, 0, 0]], dtype=np.float32))
        np.testing.assert_array_equal(faulty, clean[[0, 3]])
        np.testing.assert_array_equal(completed[-1], [1.5, 0, 0, 0])
        self.assertEqual(len(completed), 3)

    def test_rejects_injected_fault_point(self):
        with self.assertRaises(ValueError):
            retained_clean_mask(np.array([[1, 2, 3]], dtype=np.float32),
                                np.array([[1, 2, 4]], dtype=np.float32))

    def test_sector_preserves_raw_indices(self):
        clean = np.array([[1, 0, 0, 0.1], [-1, 0, 0, 0.2],
                          [2, 1, 0, 0.3]], dtype=np.float32)
        np.testing.assert_array_equal(
            radar_sector_indices(clean, np.eye(4)), [0, 2])


if __name__ == "__main__":
    unittest.main()
