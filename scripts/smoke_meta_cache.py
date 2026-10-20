"""Compare two cached frames with the uncached model and supervision path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.radar_lidar_crossmodal_v1.meta_cache import (
    MetaOccupancyCache, MetaOccupancyDataset, cache_manifest,
    collate_meta_cache, prepare_meta_batch,
)
from models.radar_lidar_crossmodal_v1.meta_occupancy import (
    MetaOccupancyConfig, candidate_support, meta_teacher_step,
)
from models.radar_lidar_crossmodal_v1.pipeline_data import collate_stage1, paired_dataset
from models.radar_lidar_crossmodal_v1.student import RadarOnlyStudent, StudentConfig
from models.radar_lidar_crossmodal_v1.student_loss import student_targets
from models.radar_lidar_crossmodal_v1.teacher import CrossModalTeacher, TeacherConfig
from models.radar_lidar_crossmodal_v1.train_meta_occupancy import _student_step


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--radar-encoder", type=Path, required=True)
    parser.add_argument("--lidar-encoder", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/crossmodal_meta_occupancy.json"))
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-gradients", action="store_true",
                        help="Run one teacher quiz and student step, checking encoder gradients")
    args = parser.parse_args()
    config = MetaOccupancyConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    raw_dataset = paired_dataset(args.vod_root, "train", args.radar_variant,
                                 data_cache_root=args.data_cache_root)
    cache = MetaOccupancyCache(
        args.cache_root / "train",
        cache_manifest(split="train",
                       frame_ids=[frame.frame_id for frame in raw_dataset.frames],
                       vod_root=args.vod_root, radar_variant=args.radar_variant,
                       radar_encoder=args.radar_encoder, lidar_encoder=args.lidar_encoder,
                       radius_m=config.radius_m, max_candidates=config.max_candidates))
    joined = MetaOccupancyDataset(raw_dataset, cache)
    student = RadarOnlyStudent(str(args.radar_encoder),
                               StudentConfig(positive_weight=1, dice_weight=0,
                                             lambda_teacher_evidence=0,
                                             decoder_from_conditioned=True),
                               trainable_encoder=True).to(args.device).eval()
    teacher = CrossModalTeacher(
        str(args.radar_encoder), str(args.lidar_encoder),
        TeacherConfig(radius_m=config.radius_m, max_candidates=config.max_candidates,
                      sigma_m=config.teacher_sigma_m, object_weight=0,
                      geometry_weight=0, contrastive_weight=0,
                      fusion_weight=0), trainable_encoders=True).to(args.device).eval()
    for index in range(2):
        raw = collate_stage1([raw_dataset[index]])
        raw = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value
               for key, value in raw.items()}
        fixed = prepare_meta_batch(collate_meta_cache([joined[index]]), args.device)
        online_student = student(raw["radar"], raw["radar_valid"])
        cached_student = student(fixed["radar"], fixed["radar_valid"])
        assert torch.equal(online_student.confidence_logits.coords,
                           cached_student.confidence_logits.coords)
        torch.testing.assert_close(online_student.conditioning.features,
                                   cached_student.conditioning.features)
        torch.testing.assert_close(online_student.confidence_logits.features,
                                   cached_student.confidence_logits.features)
        assert torch.equal(cached_student.confidence_logits.coords,
                           fixed["candidate_coords"])
        targets = student_targets(online_student, raw["clean_lidar"],
                                  raw["clean_lidar_valid"])
        assert torch.equal(targets.occupied, fixed["targets"].occupied)
        assert torch.equal(targets.known_free, fixed["targets"].known_free)
        support = candidate_support(online_student.confidence_logits.coords,
                                    raw["radar"], raw["radar_valid"], config.radius_m)
        assert torch.equal(support, fixed["support"])
        online_teacher = teacher(raw["radar"], raw["radar_valid"],
                                 raw["clean_lidar"], raw["clean_lidar_valid"])
        cached_teacher = teacher(fixed["radar"], fixed["radar_valid"],
                                 fixed["clean_lidar"], fixed["clean_lidar_valid"],
                                 neighbors=fixed["neighbors"])
        assert torch.equal(cached_teacher.radar_frozen.coords, fixed["radar_coords"])
        assert torch.equal(cached_teacher.lidar_frozen.coords, fixed["lidar_coords"])
        torch.testing.assert_close(online_teacher.fused.features,
                                   cached_teacher.fused.features)
        print(f"Frame {raw['frame_id'][0]}: cached and uncached paths agree", flush=True)
    if args.check_gradients:
        paired = prepare_meta_batch(collate_meta_cache([joined[0]]), args.device)
        quiz = prepare_meta_batch(collate_meta_cache([joined[1]]), args.device)
        teacher.train()
        student.train()
        teacher_optimizer = torch.optim.SGD(teacher.parameters(), lr=0)
        student_optimizer = torch.optim.SGD(student.parameters(), lr=0)
        with torch.enable_grad():
            meta = meta_teacher_step(teacher, student, paired, quiz,
                                     teacher_optimizer, config)
            if meta.get("skipped"):
                raise RuntimeError(f"Teacher gradient smoke skipped: {meta['reason']}")
            for name, encoder in (("teacher radar", teacher.radar_encoder),
                                  ("teacher LiDAR", teacher.lidar_encoder)):
                norm = sum(float(parameter.grad.abs().sum())
                           for parameter in encoder.parameters()
                           if parameter.grad is not None)
                if not 0 < norm < float("inf"):
                    raise RuntimeError(f"{name} encoder received no finite quiz gradient")
            _student_step(student, teacher, paired, student_optimizer, config,
                          use_teacher=True)
            norm = sum(float(parameter.grad.abs().sum())
                       for parameter in student.radar_encoder.parameters()
                       if parameter.grad is not None)
            if not 0 < norm < float("inf"):
                raise RuntimeError("Student radar encoder received no finite map gradient")
        print("Teacher radar, teacher LiDAR, and student radar gradients are finite and nonzero",
              flush=True)


if __name__ == "__main__":
    main()
