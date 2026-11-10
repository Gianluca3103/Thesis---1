"""Exercise cache provenance, a training epoch, and matched threshold evaluation."""

import json
import sys

import torch

from models.radar_lidar_crossmodal_v1.filtered_radar_cache import (
    FilteredRadarCache, filtered_frame)
from models.radar_lidar_crossmodal_v1.radar_dpcc_cache import RadarDPCCCache
from models.radar_lidar_crossmodal_v1.train_radar_dpcc import main as train_main
from scripts.compare_radar_dpcc import main as compare_main
from scripts.evaluate_radar_dpcc import main as evaluate_main


def test_small_end_to_end_radar_only_run(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    filtered = tmp_path / "filtered"
    geometry = tmp_path / "geometry"
    output = tmp_path / "model"
    for split in ("train", "val", "test"):
        (raw / split / "radar").mkdir(parents=True)
        ids = [f"{split}_0", f"{split}_1"]
        (raw / split / "manifest.json").write_text(
            json.dumps({"split": split, "frame_ids": ids}), encoding="utf-8")
        for frame_id in ids:
            points = torch.zeros((32, 7), dtype=torch.float32)
            index = torch.arange(32)
            points[:, 0] = (index % 8) * 0.2
            points[:, 1] = torch.div(index, 8, rounding_mode="floor") * 0.2
            points[:, 2] = 0.1
            points[:, 3] = 1.0
            torch.save({"frame_id": frame_id, "points": points},
                       raw / split / "radar" / f"{frame_id}.pt")
        filtered_cache = FilteredRadarCache(filtered, raw, split, create=True)
        for frame_id in ids:
            filtered_cache.write(filtered_frame(raw, split, frame_id))
        filtered_cache.check_complete()
        radar_geometry = RadarDPCCCache(filtered, raw, geometry, split, create=True)
        for i in range(len(radar_geometry)):
            radar_geometry.write(radar_geometry.build(i))
        radar_geometry.check_complete()

    monkeypatch.setattr(sys, "argv", ["train_radar_dpcc", "--raw-root", str(raw),
                                    "--filtered-root", str(filtered),
                                    "--geometry-root", str(geometry),
                                    "--output-root", str(output), "--epochs", "1",
                                    "--batch-size", "2", "--workers", "0",
                                    "--width", "16", "--device", "cpu"])
    train_main()
    assert (output / "best.ckpt").is_file()
    assert (output / "last.ckpt").is_file()
    record = json.loads((output / "history.jsonl").read_text().splitlines()[0])
    assert record["val"]["frames"] == 2
    assert record["val"]["target_points"] == 64

    report = tmp_path / "sweep.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_radar_dpcc", "--raw-root", str(raw),
                                    "--filtered-root", str(filtered),
                                    "--geometry-root", str(geometry),
                                    "--checkpoint", str(output / "best.ckpt"),
                                    "--output-json", str(report),
                                    "--batch-size", "2", "--workers", "0",
                                    "--thresholds", "0.3", "0.5", "0.7",
                                    "--device", "cpu"])
    evaluate_main()
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["frames"] == 2
    assert len(saved["rows"]) == 3
    assert saved["best_point_f1_threshold"] in (0.3, 0.5, 0.7)

    test_report = tmp_path / "test.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_radar_dpcc", "--raw-root", str(raw),
                                    "--filtered-root", str(filtered),
                                    "--geometry-root", str(geometry),
                                    "--checkpoint", str(output / "best.ckpt"),
                                    "--split", "test", "--selected-validation", str(report),
                                    "--output-json", str(test_report),
                                    "--batch-size", "2", "--workers", "0", "--device", "cpu"])
    evaluate_main()
    assert json.loads(test_report.read_text())["frames"] == 2

    monkeypatch.setattr(sys, "argv", ["compare_radar_dpcc", "--old", str(report),
                                    "--dpcc", str(report)])
    compare_main()
