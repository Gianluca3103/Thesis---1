"""Automatic, separate training/validation and correspondence diagnostics."""

from __future__ import annotations

import json
from pathlib import Path


def _series(records, split, key):
    return [row.get(split, {}).get(key) for row in records]


def update_plots(output_root: Path, phase: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Automatic overfitting plots require matplotlib") from exc
    history = output_root / "metrics.jsonl"
    if not history.is_file():
        return
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        return
    plots = output_root / "plots"
    plots.mkdir(exist_ok=True)
    epochs = [row["epoch"] for row in rows]

    def plot(name, keys, splits=("train", "val")):
        figure, axes = plt.subplots(figsize=(8, 4.5))
        drew = False
        for key in keys:
            for split in splits:
                values = _series(rows, split, key)
                if any(value is not None for value in values):
                    axes.plot(epochs, values, marker=".", label=f"{split}/{key}")
                    drew = True
        if drew:
            axes.set_xlabel("Epoch")
            axes.set_ylabel(name.replace("_", " "))
            axes.grid(alpha=0.25)
            axes.legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(plots / f"{name}.png", dpi=140)
        plt.close(figure)

    plot("total_loss", ["loss_total"])
    if phase == "teacher":
        plot("component_losses", ["loss_object", "loss_geometry", "loss_contrastive", "loss_fusion"])
        plot("object_recall", ["object_r1", "object_r3", "object_r5"], ("val",))
        plot("fusion_mae", ["fusion_mae"], ("val",))
        plot("cosine_alignment", ["positive_cosine", "negative_cosine", "cosine_gap"], ("val",))
    elif phase == "student":
        plot("student_losses", ["loss_distillation", "loss_spatial", "loss_bce", "loss_dice"])
        plot("radar_supported_iou", ["radar_supported_iou_known", "radar_supported_iou_conventional"], ("val",))
        plot("candidate_coverage", ["candidate_coverage"], ("val",))
    elif phase == "pretrain":
        plot("eligible_sites", ["eligible_sites"])
    else:
        raise ValueError(phase)
