"""Report actual parameter counts after the sparse backends and weights exist."""

import argparse
import json

from .teacher import BottleneckTeacherCore, TeacherConfig


def sizes(model):
    return {name: {"all": sum(p.numel() for p in module.parameters()),
                   "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}
            for name, module in model.named_children()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = {"teacher_core_without_weights": sizes(BottleneckTeacherCore(TeacherConfig()))}
    if args.teacher_checkpoint:
        from .train_teacher import load_teacher
        model, _ = load_teacher(args.teacher_checkpoint, args.device)
        result["teacher"] = sizes(model)
        result["teacher_core"] = sizes(model.core)
    if args.student_checkpoint:
        from .train_student import load_student
        model, _ = load_student(args.student_checkpoint, args.device)
        result["student"] = sizes(model)
        result["student_decoder"] = sizes(model.confidence_decoder)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
