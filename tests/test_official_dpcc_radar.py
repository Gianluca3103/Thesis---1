"""Validate radar-only adaptations and metric parity with the earlier AE."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.radar_self_autoencoder import reconstruction_counts as old_counts
from models.radar_lidar_stage1.sparse import voxelize
from scripts.official_dpcc_radar_data import (FILTER_VERSION, GRID_MAX, GRID_MIN,
                                              GRID_SIZE, MIN_INPUT_POINTS,
                                              FilteredFrames, RadarScale,
                                              reconstruction_counts, summarize_counts)
from scripts.patch_official_dpcc_radar import COMPAT_CPP, UPSTREAM_REV, adapt
from scripts.train_radar_dpcc_official import ModelConfig, reset_encoder_neighbors


def test_pinned_upstream_changes_only_radar_channels(tmp_path: Path):
    source = Path(__file__).parents[2] / "dpcc-upstream" / "models"
    if not source.exists():
        return  # CI may not have the optional external source checkout.
    for name in ("layer.py", "autoencoder.py", "utils.py"):
        dest = tmp_path / "models" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((source / name).read_bytes())
    for name in COMPAT_CPP:
        dest = tmp_path / "models/pointops/src" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((source / "pointops/src" / name).read_bytes())
    dest = tmp_path / "models/Chamfer3D/chamfer3D.cu"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes((source / "Chamfer3D/chamfer3D.cu").read_bytes())
    with patch("scripts.patch_official_dpcc_radar.subprocess.check_output",
               return_value=UPSTREAM_REV):
        first = adapt(tmp_path)
        second = adapt(tmp_path)
    assert first == second
    assert "getattr(args, 'radar_attribute_dim', 3)" in (tmp_path / "models/layer.py").read_text()
    assert "return decompressed_xyzs, pred_normals, loss" in (
        tmp_path / "models/autoencoder.py").read_text()
    assert "#include <THC/THC.h>" not in (
        tmp_path / "models/pointops/src" / COMPAT_CPP[0]).read_text()
    assert "import open3d" not in (tmp_path / "models/utils.py").read_text()


def test_paper_config_except_radar_width():
    c = ModelConfig()
    assert c.in_fdim == 7 and c.radar_attribute_dim == 4
    assert c.k == 16 and c.layer_num == 3 and c.dim == 8
    assert c.downsample_rate == (1 / 3,) * 3
    assert c.max_upsample_num == (8, 8, 8)
    assert c.bpp_lambda == 1e-3 and c.chamfer_coe == 0.1
    assert c.lr == 1e-3 and c.aux_lr == 1e-3


def test_professor_launcher_builds_compressai_from_source():
    launcher = (Path(__file__).parents[1] / "scripts/run_radar_dpcc_official_professor.sh").read_text()
    assert "--no-binary=compressai" in launcher
    assert "--only-binary=:all:" not in launcher
    assert "compressai==1.2.8" in launcher


def test_professor_launcher_selects_matching_cuda_and_installer_is_toolkit_only():
    root = Path(__file__).parents[1] / "scripts"
    launcher = (root / "run_radar_dpcc_official_professor.sh").read_text()
    installer = (root / "install_cuda121_for_dpcc.sh").read_text()
    assert '"$BASE/cuda-12.1"' in launcher
    assert '"$version" == "$torch_cuda"' in launcher
    assert "--toolkitpath=" in installer and "--tmpdir=" in installer
    assert "--driver" not in installer and "sudo" not in installer.split("sh \"$INSTALLER\"")[1]
    assert "2f0a4127bf797bf4eab0be2a547cb8d0" in installer


def test_normalization_inverts_training_values():
    scale = RadarScale([-30, -15, -10, 0], [50, 20, 10, 19])
    points = torch.tensor([[1.2, -4.1, 0.8, 12, -2, 1, 8],
                           [-20.0, 30.0, -1.4, -30, 20, -10, 19]], dtype=torch.float32)
    restored = scale.decode(scale.encode(points))
    assert torch.allclose(restored, points, atol=1e-5)


def test_small_nonempty_filtered_frame_is_not_rejected(tmp_path: Path):
    assert MIN_INPUT_POINTS == 14
    assert [round(14 / 3), round(round(14 / 3) / 3),
            round(round(round(14 / 3) / 3) / 3)] == [5, 2, 1]
    root = tmp_path / "train"
    root.mkdir()
    manifest = {"version": FILTER_VERSION, "split": "train", "frame_ids": ["00544"],
                "min_distinct_radar_voxels": 12, "radius_m": 1.6,
                "grid_minimum_xyz": GRID_MIN.tolist(),
                "grid_maximum_xyz": GRID_MAX.tolist(),
                "grid_size_xyz": GRID_SIZE.tolist()}
    (root / "manifest.json").write_text(json.dumps(manifest))
    path = root / "00544.pt"
    torch.save({"version": FILTER_VERSION, "frame_id": "00544",
                "points": torch.ones((46, 7))}, path)
    frames = FilteredFrames(tmp_path, "train")
    assert frames.load(0).shape == (46, 7)
    torch.save({"version": FILTER_VERSION, "frame_id": "00544",
                "points": torch.ones((13, 7))}, path)
    with pytest.raises(ValueError, match="require >= 14"):
        frames.load(0)


def test_small_frame_does_not_permanently_shrink_neighbor_count():
    layers = [SimpleNamespace(k=2), SimpleNamespace(k=5), SimpleNamespace(k=15)]
    model = SimpleNamespace(args=SimpleNamespace(k=16),
                            encoder=SimpleNamespace(encoder_layers=layers))
    reset_encoder_neighbors(model)
    assert [layer.k for layer in layers] == [16, 16, 16]


def test_same_counts_as_previous_radar_evaluator():
    target = np.array([[0.12, 0.04, 0.05, 10, 1, 1.1, 0],
                       [0.42, 0.04, 0.05, 12, 2, 2.1, 1],
                       [1.12, 0.04, 0.05, 14, 3, 3.1, 2]], dtype=np.float32)
    predicted = np.array([[0.13, 0.05, 0.05, 9, 1.5, 1.1, 0],
                          [0.45, 0.05, 0.05, 13, 2.5, 2.1, 1],
                          [1.82, 0.05, 0.05, 14, 3, 3.1, 2]], dtype=np.float32)
    pt = torch.from_numpy(predicted)
    gt = torch.from_numpy(target)
    voxels, _, _, _ = voxelize(pt[None], torch.ones((1, len(pt)), dtype=torch.bool), GRID)
    old = old_counts(torch.cat((torch.zeros((len(pt), 1)), pt), 1), voxels,
                     gt[None], torch.ones((1, len(gt)), dtype=torch.bool))
    new = reconstruction_counts(predicted, target)
    for key in old:
        assert np.isclose(new[key], old[key]), (key, new[key], old[key])
    summary = summarize_counts(new)
    assert np.isclose(summary["voxel_iou"], new["voxel_tp"] / (
        new["predicted_voxels"] + new["target_voxels"] - new["voxel_tp"]))
