import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Fault_Localization_Model.vod_dataset.radar_accumulation import (
    RadarTemporalFilterConfig,
    accumulate_vod_radar_scans,
    filter_accumulated_radar_points,
    radar_current_from_source,
    radar_current_from_official_previous,
)
from scripts.generate_vod_accumulated_radar import _histories, _source_to_current


def _write_pose(path: Path, transform: np.ndarray) -> None:
    path.write_text(json.dumps({"odomToCamera": transform.reshape(-1).tolist()}))


def _write_calibration(path: Path, transform: np.ndarray) -> None:
    values = " ".join(str(value) for value in transform[:3].reshape(-1))
    path.write_text(f"Tr_velo_to_cam: {values}\n")


class VoDRadarAccumulationTests(unittest.TestCase):
    def test_validity_filter_removes_out_of_range_and_non_finite_points(self):
        points = np.asarray(
            [
                [2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 8.0, 5.0, 0.0, 0.0, 0.0],
                [np.nan, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        filtered, stats = filter_accumulated_radar_points(
            points,
            RadarTemporalFilterConfig(),
        )
        self.assertEqual(filtered.shape, (1, 7))
        self.assertEqual(stats["validity_rejected"], 3)

    def test_temporal_filter_keeps_supported_history_and_current_scan(self):
        points = np.asarray(
            [
                [10.0, 0.0, 0.0, 5.0, 0.0, 0.0, -2.0],
                [10.2, 0.1, 0.0, 5.0, 0.0, 0.0, -1.0],
                [20.0, 0.0, 0.0, 5.0, 0.0, 0.0, -2.0],
                [30.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        filtered, stats = filter_accumulated_radar_points(
            points,
            RadarTemporalFilterConfig(temporal_radius_m=0.5),
        )
        np.testing.assert_allclose(filtered[:, 0], [10.0, 10.2, 30.0])
        self.assertEqual(stats["temporal_rejected"], 1)

    def test_same_scan_neighbors_do_not_count_as_temporal_support(self):
        points = np.asarray(
            [
                [10.0, 0.0, 0.0, 5.0, 0.0, 0.0, -1.0],
                [10.1, 0.0, 0.0, 5.0, 0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        filtered, _ = filter_accumulated_radar_points(
            points,
            RadarTemporalFilterConfig(temporal_radius_m=0.5),
        )
        self.assertEqual(filtered.shape, (0, 7))

    def test_motion_gate_discards_old_movers_without_warping_returns(self):
        points = np.asarray([
            [10, 0, 0, 5, 0, 0.05, -19],  # stationary long history
            [11, 0, 0, 5, 0, 2.0, -19],   # stale moving return
            [12, 0, 0, 5, 0, 2.0, -1],    # recent moving return
            [13, 0, 0, 5, 0, 2.0, 0],     # current moving return
            [14, 0, 0, 5, 0, 0.0, -19],   # radial Doppler cannot reveal lateral motion
        ], dtype=np.float32)
        filtered, stats = filter_accumulated_radar_points(
            points,
            RadarTemporalFilterConfig(
                moving_velocity_threshold_mps=1.0,
                moving_max_age_scans=1,
            ),
        )
        np.testing.assert_array_equal(filtered, points[[0, 2, 3, 4]])
        self.assertEqual(stats["motion_rejected"], 1)
        self.assertEqual(stats["validity_rejected"], 0)

    def test_motion_gate_and_temporal_filter_can_run_together(self):
        points = np.asarray([
            [10.0, 0, 0, 5, 0, 0.1, -2],
            [11.0, 0, 0, 5, 0, 2.0, -5],  # rejected before temporal filtering
            [10.1, 0, 0, 5, 0, 0.1, -1],
            [10.2, 0, 0, 5, 0, 0.1, 0],
        ], dtype=np.float32)
        filtered, stats = filter_accumulated_radar_points(
            points, RadarTemporalFilterConfig(
                moving_velocity_threshold_mps=1.0,
                moving_max_age_scans=1,
                temporal_radius_m=0.5,
                temporal_min_scans=2,
            ),
        )
        np.testing.assert_array_equal(filtered, points[[0, 2, 3]])
        self.assertEqual(stats["motion_rejected"], 1)
        self.assertEqual(stats["temporal_rejected"], 0)

    def test_motion_gate_rejects_invalid_settings(self):
        with self.assertRaises(ValueError):
            RadarTemporalFilterConfig(moving_velocity_threshold_mps=0).validate()
        with self.assertRaises(ValueError):
            RadarTemporalFilterConfig(moving_max_age_scans=-1).validate()

    def test_source_scan_is_motion_compensated_into_current_radar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_pose = root / "source_pose.json"
            current_pose = root / "current_pose.json"
            source_calibration = root / "source_calib.txt"
            current_calibration = root / "current_calib.txt"
            source = np.eye(4)
            source[0, 3] = 2.0
            current = np.eye(4)
            calibration = np.eye(4)
            _write_pose(source_pose, source)
            _write_pose(current_pose, current)
            _write_calibration(source_calibration, calibration)
            _write_calibration(current_calibration, calibration)

            transform = radar_current_from_source(
                source_pose,
                current_pose,
                source_calibration,
                current_calibration,
            )
            np.testing.assert_allclose(transform[:3, 3], [2.0, 0.0, 0.0])

    def test_accumulation_preserves_features_and_sets_time_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            radar_paths = []
            pose_paths = []
            calibration_paths = []
            for index in range(2):
                radar_path = root / f"radar_{index}.bin"
                pose_path = root / f"pose_{index}.json"
                calibration_path = root / f"calib_{index}.txt"
                np.asarray(
                    [[1.0, 2.0, 3.0, 4.0 + index, 5.0, 6.0, 0.0]],
                    dtype=np.float32,
                ).tofile(radar_path)
                _write_pose(pose_path, np.eye(4))
                _write_calibration(calibration_path, np.eye(4))
                radar_paths.append(radar_path)
                pose_paths.append(pose_path)
                calibration_paths.append(calibration_path)

            output = accumulate_vod_radar_scans(
                radar_paths, pose_paths, calibration_paths
            )
            np.testing.assert_allclose(output[:, 3:6], [[4, 5, 6], [5, 5, 6]])
            np.testing.assert_allclose(output[:, 6], [-1, 0])

    def test_official_alignment_recovers_rigid_transform_and_preserves_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.bin"
            official = root / "official.bin"
            source = np.asarray([
                [1, 2, 0, 4, 2, 1, 0],
                [3, 1, 1, 5, 3, 2, 0],
                [2, 4, 2, 6, 4, 3, 0],
                [5, 3, 3, 7, 5, 4, 0],
            ], dtype=np.float32)
            source.tofile(raw)
            shifted = source.copy()
            shifted[:, :3] += [0.3, -0.4, 0.2]
            shifted[:, 6] = -1
            np.concatenate((shifted, source), axis=0).tofile(official)
            transform = radar_current_from_official_previous(raw, official)
            np.testing.assert_allclose(transform[:3, 3], [0.3, -0.4, 0.2], atol=1e-6)

    def test_official_scene_boundary_and_transform_composition(self):
        steps = {12: np.eye(4), 13: np.eye(4)}
        steps[12][0, 3] = 1.0
        steps[13][0, 3] = 2.0
        histories = _histories(Path("unused"), [10, 11, 12, 13], 20, 5.0,
                               "training", steps)
        self.assertEqual(histories[11], [11])
        self.assertEqual(histories[13], [11, 12, 13])
        transforms = _source_to_current(histories[13], steps)
        np.testing.assert_allclose([item[0, 3] for item in transforms], [3, 2, 0])


if __name__ == "__main__":
    unittest.main()
