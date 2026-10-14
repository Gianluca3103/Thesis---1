"""Held-out evaluation for the independently checkpointed teacher or student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from .pipeline_data import collate_stage1, paired_dataset
from .train_teacher import _epoch as evaluate_teacher_epoch, load_teacher
from .train_student import _epoch as evaluate_student_epoch, load_student


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("teacher", "student"))
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.phase == "teacher":
        teacher, saved = load_teacher(args.checkpoint, args.device)
        variant = saved["radar_variant"]
    else:
        student, saved = load_student(args.checkpoint, args.device)
        teacher, _ = load_teacher(saved["teacher_checkpoint"], args.device)
        variant = saved["radar_variant"]
    dataset = paired_dataset(args.vod_root, args.split, variant, limit=args.limit)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers,
                        collate_fn=collate_stage1)
    if args.phase == "teacher":
        report = evaluate_teacher_epoch(teacher, dataset, loader, args.device)
    else:
        report = evaluate_student_epoch(student, teacher, dataset, loader, args.device)
    report.update(phase=args.phase, split=args.split, checkpoint=str(args.checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
