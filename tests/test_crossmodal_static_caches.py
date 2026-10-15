"""Fixed input and frozen teacher cache contracts for the no-augmentation run."""

from types import SimpleNamespace

import pytest
import torch

from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.fixed_frame_cache import FixedFrameCache
from models.radar_lidar_crossmodal_v1.teacher import BottleneckTeacherCore, TeacherConfig
from models.radar_lidar_crossmodal_v1.teacher_static_cache import TeacherStaticCache
from models.radar_lidar_crossmodal_v1.train_teacher import _epoch as teacher_epoch
from models.radar_lidar_stage1.correspondence import Neighborhood
from models.radar_lidar_stage1.sparse import SparseSites


def _frame(frame_id, root):
    return SimpleNamespace(frame_id=frame_id, split="train",
                           radar_variant="radar-test",
                           lidar_path=root / "lidar/training/velodyne" / f"{frame_id}.bin")


def _sites(coords, features):
    return SparseSites(torch.tensor(coords, dtype=torch.long), features,
                       GRID.scale_shape(8), 8)


def test_fixed_frames_preserve_inputs_and_support_radar_only(tmp_path):
    dataset = SimpleNamespace(frames=[_frame("00001", tmp_path)],
                              augmentation_config=SimpleNamespace(enabled=False))
    cache = FixedFrameCache(tmp_path / "cache", dataset, create=True)
    radar = torch.tensor([[1., 2., 3., 4., 5., 6., 7.]])
    lidar = torch.tensor([[1., 2., 3., 4.]])
    cache.write({"frame_id": "00001", "radar": radar, "clean_lidar": lidar})
    assert torch.equal(cache[0]["radar"], radar)
    assert torch.equal(cache[0]["clean_lidar"], lidar)
    only_radar = FixedFrameCache(tmp_path / "cache", dataset, include_clean=False)
    assert "clean_lidar" not in only_radar[0]
    only_radar.check_complete()
    with pytest.raises(ValueError, match="augmentation"):
        FixedFrameCache(tmp_path / "other",
                        SimpleNamespace(frames=dataset.frames,
                                        augmentation_config=SimpleNamespace(enabled=True)),
                        create=True)


def test_teacher_static_cache_rebases_neighbors_and_object_ids(tmp_path):
    cache = TeacherStaticCache(tmp_path, split="train", radar_variant="radar-test",
                               radar_sha256="radar-hash", lidar_sha256="lidar-hash",
                               radius_m=6., max_candidates=2, create=True)
    for frame_id, value in (("00001", 1.), ("00002", 2.)):
        radar = _sites([[0, 0, 1, 1]], torch.full((1, 64), value))
        lidar = _sites([[0, 0, 1, 1]], torch.full((1, 64), value + 1))
        neighbors = Neighborhood(torch.tensor([[0, -1]]),
                                 torch.tensor([[0., float("inf")]]))
        cache.write(frame_id, radar, lidar, neighbors, torch.tensor([7]),
                    torch.tensor([7]), {7: "Car"})
    radar, lidar, neighbors, radar_ids, lidar_ids, classes = cache.load_batch(
        ["00001", "00002"], "cpu")
    assert radar.coords[:, 0].tolist() == [0, 1]
    assert lidar.coords[:, 0].tolist() == [0, 1]
    assert neighbors.indices.tolist() == [[0, -1], [1, -1]]
    assert radar_ids.tolist() == [7, 1_000_007]
    assert lidar_ids.tolist() == [7, 1_000_007]
    assert classes == {7: "Car", 1_000_007: "Car"}
    core = BottleneckTeacherCore(TeacherConfig(max_candidates=2))
    output = core(radar, lidar, neighbors)
    output.fused.features.square().sum().backward()
    assert core.query.weight.grad is not None
    assert torch.isfinite(core.query.weight.grad).all()

    class CachedTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = TeacherConfig(max_candidates=2)
            self.core = BottleneckTeacherCore(self.config)

        def forward(self, *args):
            raise AssertionError("Cached teacher training must not run the encoders")

    class Dataset:
        def set_epoch(self, epoch):
            self.epoch = epoch

    model = CachedTeacher()
    dataset = Dataset()
    optimizer = torch.optim.SGD(model.core.parameters(), lr=0.01)
    report = teacher_epoch(model, dataset,
                           [{"frame_id": ["00001", "00002"]}],
                           "cpu", optimizer=optimizer, epoch=3,
                           static_cache=cache)
    assert dataset.epoch == 3
    assert report["frames"] == 2
    assert report["radar_active_sites"] == 2
    with pytest.raises(ValueError, match="manifest"):
        TeacherStaticCache(tmp_path, split="train", radar_variant="radar-test",
                           radar_sha256="changed", lidar_sha256="lidar-hash",
                           radius_m=6., max_candidates=2)
