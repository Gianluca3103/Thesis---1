"""Summarize matched full-validation results for head-only versus joint training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


METRICS = ("proposal_utility", "distinct_correct_voxels",
           "verified_proposal_precision", "false_positive_known_free",
           "unknown_selected_fraction", "selected_sites", "candidate_sites",
           "loss_pointer", "loss_distillation")


def _records(root: Path, arm: str):
    path = root / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    full = [row for row in records if row["validation_scope"] == "full"]
    if not full or any(row["arm"] != arm for row in full):
        raise ValueError(f"No full-validation {arm} records in {path}")
    return full


def _selected(row):
    return {"epoch": row["epoch"], "frames": row["val"]["frames"],
            **{key: row["val"].get(key) for key in METRICS}}


def compare(control_root: Path, joint_root: Path) -> dict:
    control = _records(control_root, "control")
    joint = _records(joint_root, "joint")
    a = torch.load(control_root / "last.ckpt", map_location="cpu", weights_only=False)
    b = torch.load(joint_root / "last.ckpt", map_location="cpu", weights_only=False)
    for key in ("initial_surface_sha256", "initial_feature_sha256",
                "teacher_sha256", "seed"):
        if a[key] != b[key]:
            raise ValueError(f"Ablation arms differ in {key}")
    if a["distillation_weight"] != 0 or not a["freeze_feature_student"]:
        raise ValueError("Control is not head-only")
    if b["distillation_weight"] <= 0 or b["freeze_feature_student"]:
        raise ValueError("Joint arm does not train the feature student")
    control_by_epoch = {row["epoch"]: row for row in control}
    joint_by_epoch = {row["epoch"]: row for row in joint}
    shared = sorted(control_by_epoch.keys() & joint_by_epoch.keys())
    if not shared:
        raise ValueError("Ablation arms have no matched full-validation epoch")
    final_epoch = shared[-1]
    ca, jb = (_selected(records[final_epoch]) for records in
              (control_by_epoch, joint_by_epoch))
    if ca["frames"] != jb["frames"]:
        raise ValueError("Ablation arms evaluated different frame counts")
    delta = {key: jb[key] - ca[key] if ca[key] is not None and jb[key] is not None
             else None for key in METRICS}
    best_control = _selected(max(control, key=lambda row: row["val"]["proposal_utility"]))
    best_joint = _selected(max(joint, key=lambda row: row["val"]["proposal_utility"]))
    return {"initial_surface_sha256": a["initial_surface_sha256"],
            "teacher_sha256": a["teacher_sha256"],
            "matched_final_epoch": final_epoch,
            "matched_final_control": ca, "matched_final_joint": jb,
            "matched_final_joint_minus_control": delta,
            "best_full_control": best_control, "best_full_joint": best_joint}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.control_root, args.joint_root)
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
