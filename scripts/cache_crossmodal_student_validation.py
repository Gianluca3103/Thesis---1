"""Build fixed train or validation supervision before radar-only student training."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import collate_stage1, paired_dataset
from models.radar_lidar_crossmodal_v1.student import RadarOnlyStudent
from models.radar_lidar_crossmodal_v1.student_loss import (
    student_targets, teacher_correspondence_evidence,
)
from models.radar_lidar_crossmodal_v1.student_validation_cache import StudentValidationCache
from models.radar_lidar_crossmodal_v1.train_student import load_student
from models.radar_lidar_crossmodal_v1.train_teacher import _hash, _move, load_teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--minimum-free-gib", type=float, default=5.)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        parser.error("Cache construction needs the same CUDA sparse runtime as training")
    teacher, source = load_teacher(args.teacher_checkpoint, args.device)
    if source["radar_variant"] != args.radar_variant:
        raise ValueError("Teacher and cache radar variants differ")
    if args.student_checkpoint:
        student, saved = load_student(args.student_checkpoint, args.device)
        if saved["radar_checkpoint"] != source["radar_checkpoint"]:
            raise ValueError("Student and teacher use different frozen radar encoders")
    else:
        student = RadarOnlyStudent(source["radar_checkpoint"]).to(args.device).eval()
    cache = StudentValidationCache(args.cache_root, _hash(args.teacher_checkpoint),
                                   args.radar_variant, split=args.split, create=True)
    limit = args.train_limit if args.split == "train" else args.val_limit
    dataset = paired_dataset(args.vod_root, args.split, args.radar_variant, limit=limit)
    missing_indices = []
    reused = 0
    for index, frame in enumerate(dataset.frames):
        path = cache.path(frame.frame_id)
        if path.is_file():
            cache._read(frame.frame_id)
            reused += 1
        else:
            missing_indices.append(index)
    loader = DataLoader(Subset(dataset, missing_indices), batch_size=1,
                        num_workers=args.num_workers,
                        collate_fn=collate_stage1, pin_memory=True)
    measured_bytes = []
    built = 0
    for raw in tqdm(loader, desc=f"Student {args.split} cache", unit="frame", dynamic_ncols=True):
        frame_id = raw["frame_id"][0]
        batch = _move(raw, args.device)
        with torch.no_grad():
            teacher_output = teacher(batch["radar"], batch["radar_valid"],
                                     batch["clean_lidar"], batch["clean_lidar_valid"])
            output = student(batch["radar"], batch["radar_valid"])
            targets = student_targets(output, batch["clean_lidar"],
                                      batch["clean_lidar_valid"])
            evidence = teacher_correspondence_evidence(
                teacher_output, output.confidence_logits)
        measured_bytes.append(cache.write(frame_id, output, teacher_output,
                                          targets, evidence))
        built += 1
        if len(measured_bytes) == min(8, len(missing_indices)):
            average = sum(measured_bytes) / len(measured_bytes)
            remaining = len(missing_indices) - built
            free = shutil.disk_usage(args.cache_root).free
            required = average * remaining * 1.2 + args.minimum_free_gib * 1024**3
            print(f"Cache pilot: {average / 1024**2:.1f} MiB/frame, "
                  f"estimated remaining {average * remaining / 1024**3:.1f} GiB, "
                  f"free {free / 1024**3:.1f} GiB", flush=True)
            if required > free:
                raise OSError(f"Insufficient space for the complete {args.split} cache; "
                              "existing entries remain reusable")
    print(f"{args.split.capitalize()} cache ready: {built} built, {reused} reused, "
          f"{len(dataset)} frames at {args.cache_root}", flush=True)


if __name__ == "__main__":
    main()
