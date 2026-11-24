"""Partial-only faulty scans preserve clean returns and exclude fog."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.mild_fault_data import (
    PARTIAL_ONLY_TYPES, PARTIAL_ONLY_VERSION, balanced_fault_assignment,
    fault_types_for_version,
)
from scripts.cache_stage2_partial_loss import _build_frame
from scripts import cache_stage2_fault_regions as region_builder


def test_partial_only_assignment_is_balanced_and_has_no_fog():
    ids = [f"{index:05d}" for index in range(101)]
    assignment = balanced_fault_assignment(
        ids, "train", 3103, fault_types=PARTIAL_ONLY_TYPES)
    counts = [list(assignment.values()).count(fault) for fault in PARTIAL_ONLY_TYPES]
    assert max(counts) - min(counts) == 1
    assert fault_types_for_version(PARTIAL_ONLY_VERSION) == PARTIAL_ONLY_TYPES
    assert all(fault.startswith("partial_loss_") for fault in assignment.values())


def test_partial_cache_contains_only_surviving_clean_xyz(tmp_path):
    points = torch.tensor(
        [[float(i + 1), float(i % 3), 0.1, float(i)] for i in range(20)])
    clean_path = tmp_path / "clean.pt"
    torch.save({"frame_id": "00001", "points": points}, clean_path)
    root = tmp_path / "faulty"
    path = root / "train" / "00001.npz"
    job = ("00001", "train", "partial_loss_40", 3103,
           clean_path, path, "clean-manifest-hash")
    assert _build_frame(job)[:3] == ("00001", "partial_loss_40", False)
    retained = load_faulty(root, "train", "00001").numpy()
    assert retained.shape == (12, 3)
    assert all(any(np.array_equal(row, clean) for clean in points[:, :3].numpy())
               for row in retained)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    assert metadata["version"] == PARTIAL_ONLY_VERSION
    assert metadata["fault_type"] == "partial_loss_40"
    assert metadata["removed_fraction_requested"] == 0.4
    assert _build_frame(job)[:3] == ("00001", "partial_loss_40", True)


def test_fault_region_cache_accepts_partial_only_family(
        tmp_path, monkeypatch):
    split = "train"
    ids = ["00001", "00002", "00003"]
    clean_root = tmp_path / "fixed" / split
    clean_folder = clean_root / "clean_lidar"
    clean_folder.mkdir(parents=True)
    (clean_root / "manifest.json").write_text("{}", encoding="utf-8")
    faulty_root = tmp_path / "faulty"
    faults = dict(zip(ids, PARTIAL_ONLY_TYPES))
    (faulty_root / split).mkdir(parents=True)
    (faulty_root / split / "manifest.json").write_text(json.dumps({
        "version": PARTIAL_ONLY_VERSION, "split": split,
        "seed": 3103,
        "radar_variant": "radar_20frames_verified_doppler_radial",
        "frame_ids": ids, "fault_types": list(PARTIAL_ONLY_TYPES),
        "frame_faults": faults,
    }), encoding="utf-8")
    for frame_id, fault in faults.items():
        points = torch.tensor(
            [[float(i + 1), float(i % 3), 0.1, float(i)] for i in range(20)])
        clean_path = clean_folder / f"{frame_id}.pt"
        torch.save({"frame_id": frame_id, "points": points}, clean_path)
        _build_frame((frame_id, split, fault, 3103, clean_path,
                      faulty_root / split / f"{frame_id}.npz", "hash"))

    class Dataset:
        frames = [SimpleNamespace(frame_id=frame_id) for frame_id in ids]

        def path(self, frame_id, kind):
            assert kind == "clean_lidar"
            return clean_folder / f"{frame_id}.pt"

    monkeypatch.setattr(region_builder, "paired_dataset", lambda *_, **__: Dataset())
    output = tmp_path / "regions"
    monkeypatch.setattr(sys, "argv", [
        "cache_stage2_fault_regions", "--vod-root", str(tmp_path),
        "--data-cache-root", str(tmp_path / "fixed"),
        "--faulty-root", str(faulty_root),
        "--output-root", str(output), "--split", split,
    ])
    region_builder.main()
    regions = json.loads((output / split / "regions.json").read_text())
    assert set(regions) == set(ids)
    assert all(region["kind"] == "azimuth_sector" for region in regions.values())
