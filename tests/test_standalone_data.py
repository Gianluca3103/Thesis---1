"""The extracted loader still discovers, calibrates, and batches VoD pairs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage1.radar_fov import (
    RADAR_HALF_FOV_DEG, RADAR_MAX_RANGE_M, crop_lidar_to_radar_fov,
)
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset


def _calibration(path: Path, x: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Tr_velo_to_cam: 1 0 0 {x} 0 1 0 0 0 0 1 0\n")


class StandaloneVoDDataTests(unittest.TestCase):
    def test_paired_radar_is_aligned_and_collated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, frame_id in (("train", "00001"), ("val", "00002")):
                image_sets = root / "lidar" / "ImageSets"
                image_sets.mkdir(parents=True, exist_ok=True)
                (image_sets / f"{split}.txt").write_text(frame_id + "\n")
                lidar_root = root / "lidar" / "training"
                radar_root = root / "radar_20frames_verified_doppler_radial" / "training"
                (lidar_root / "velodyne").mkdir(parents=True, exist_ok=True)
                (radar_root / "velodyne").mkdir(parents=True, exist_ok=True)
                np.array([[2., 0., 0., 0.5], [0., 0., 0., 0.6],
                          [2., 10., 0., 0.7]], dtype=np.float32).tofile(
                    lidar_root / "velodyne" / f"{frame_id}.bin")
                np.array([[1., 0., 0., 3., 4., 5., 0.]], dtype=np.float32).tofile(
                    radar_root / "velodyne" / f"{frame_id}.bin")
                _calibration(lidar_root / "calib" / f"{frame_id}.txt", 0.)
                # Derived radar variants inherit single-scan calibration.
                _calibration(root / "radar" / "training" / "calib" /
                             f"{frame_id}.txt", 1.)
            dataset = VoDStage1Dataset(root, "train")
            sample = dataset[0]
            self.assertEqual(sample["frame_id"], "00001")
            np.testing.assert_allclose(sample["radar"][0, :3], [2., 0., 0.])
            self.assertEqual(sample["radar"][0, 3:].tolist(), [3., 4., 5., 0.])
            batch = collate_stage1([sample])
            self.assertEqual(batch["radar"].shape, (1, 1, 7))
            self.assertTrue(batch["clean_lidar_valid"][0, 0])
            self.assertEqual(len(sample["clean_lidar"]), 3)
            cropped = VoDStage1Dataset(
                root, "train", lidar_radar_fov_deg=RADAR_HALF_FOV_DEG,
                lidar_radar_max_range_m=RADAR_MAX_RANGE_M)[0]
            np.testing.assert_allclose(cropped["clean_lidar"].numpy(),
                                       [[2., 0., 0., 0.5]])
            crossmodal = paired_dataset(
                root, "train", "radar_20frames_verified_doppler_radial")
            np.testing.assert_allclose(crossmodal[0]["clean_lidar"].numpy(),
                                       [[2., 0., 0., 0.5]])
            self.assertEqual(len(VoDStage1Dataset(root, "val")), 1)

    def test_radar_sector_uses_calibrated_origin_and_preserves_features(self):
        # Radar +X becomes LiDAR +Y after this calibration, with an offset.
        transform = np.eye(4)
        transform[:2, :2] = [[0., -1.], [1., 0.]]
        transform[:3, 3] = [3., 4., 0.]
        radar_xyz = np.array([[10., 0., 0.], [10., 30., 0.],
                              [-1., 0., 0.], [81., 0., 0.]], dtype=np.float32)
        lidar_xyz = radar_xyz @ transform[:3, :3].T + transform[:3, 3]
        lidar = np.column_stack((lidar_xyz, [11., 22., 33., 44.])).astype(np.float32)
        selected = crop_lidar_to_radar_fov(lidar, transform)
        np.testing.assert_allclose(selected, lidar[:1])


if __name__ == "__main__":
    unittest.main()
