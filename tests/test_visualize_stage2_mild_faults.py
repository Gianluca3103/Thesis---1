"""Synthetic cache contract checks for the standalone mild-fault viewer."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from models.radar_lidar_stage2.mild_fault_data import (
    MILD_FAULT_ENCODING, MILD_FAULT_TYPES, MILD_FAULT_VERSION, encode_mild_xyz,
)
from scripts.visualize_stage2_mild_faults import _removed_mask, render


class MildFaultViewerTest(unittest.TestCase):
    def test_removed_mask_respects_duplicates_and_rejects_new_points(self):
        clean = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]], dtype=np.float32)
        faulty = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        mask = _removed_mask(clean, faulty)
        self.assertEqual(mask.tolist(), [False, True, False])
        with self.assertRaisesRegex(ValueError, "absent from clean"):
            _removed_mask(clean, np.array([[9, 9, 9]], dtype=np.float32))

    def test_renders_each_fault_type_and_full_point_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mild = root / "mild"
            split = "val"
            mild_split = mild / split
            mild_split.mkdir(parents=True)
            clean_root = root / "fixed" / split / "clean_lidar"
            clean_root.mkdir(parents=True)
            ids = [f"frame{i}" for i in range(4)]
            manifest = {"version": MILD_FAULT_VERSION, "split": split,
                        "data_cache_root": str(root / "fixed"),
                        "source_manifest_sha256": "synthetic",
                        "frame_ids": ids,
                        "frame_faults": dict(zip(ids, MILD_FAULT_TYPES))}
            (mild_split / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            clean_xyz = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0],
                                  [4, 0, 0]], dtype=np.float32)
            for index, frame_id in enumerate(ids):
                clean = np.column_stack((clean_xyz, np.ones(4, dtype=np.float32)))
                torch.save({"frame_id": frame_id, "points": torch.from_numpy(clean)},
                           clean_root / f"{frame_id}.pt")
                faulty = clean_xyz[:4 - index]
                metadata = {"version": MILD_FAULT_VERSION, "encoding": MILD_FAULT_ENCODING,
                            "frame_id": frame_id, "split": split,
                            "fault_type": MILD_FAULT_TYPES[index],
                            "source_manifest_sha256": "synthetic",
                            "clean_points": 4, "retained_points": len(faulty)}
                np.savez_compressed(mild_split / f"{frame_id}.npz",
                                    points_bytes=encode_mild_xyz(faulty),
                                    metadata_json=np.asarray(json.dumps(metadata)))
            output = root / "views"
            rows = render(mild, output, split, 1, None, 3103, 3)
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["removed_points"] for row in rows], [0, 1, 2, 3])
            self.assertIn("Stage II mild LiDAR faults", (output / "index.html").read_text())
            for row in rows:
                markup = (output / row["html"]).read_text(encoding="utf-8")
                self.assertIn(row["fault_type"], markup)
                self.assertIn("Removed points", markup)
                for name in ("clean", "faulty", "removed"):
                    path = output / f'{row["stem"]}_{name}.ply'
                    self.assertTrue(path.is_file())
                    field = {"clean": "clean_points", "faulty": "retained_points",
                             "removed": "removed_points"}[name]
                    self.assertIn(f'element vertex {row[field]}',
                                  path.read_bytes().split(b"end_header")[0].decode("ascii"))


if __name__ == "__main__":
    unittest.main()
