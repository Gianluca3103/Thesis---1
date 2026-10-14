"""TensorBoard curves and trend-only overfitting warnings."""

from __future__ import annotations

import json
from pathlib import Path


def open_writer(root: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard logging requires the tensorboard package") from exc
    return SummaryWriter(log_dir=str(root / "tensorboard"))


def log_record(writer, record):
    if writer is None:
        return
    for split in ("train", "val"):
        for name, value in record[split].items():
            if isinstance(value, (float, int)) and value is not None:
                writer.add_scalar(f"{split}/{name}", value, record["epoch"])
    for name in ("loss_gap", "relative_loss_gap"):
        if record.get(name) is not None:
            writer.add_scalar(f"diagnostics/{name}", record[name], record["epoch"])
    writer.flush()


def overfitting_warning(history_path: Path) -> str | None:
    if not history_path.is_file():
        return None
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) < 3:
        return None
    recent = rows[-3:]
    if len({row["validation_scope"] for row in recent}) != 1:
        return None
    train = [row["train"]["loss_total"] for row in recent]
    val = [row["val"]["loss_total"] for row in recent]
    if any(value is None for value in (*train, *val)):
        return None
    if train[0] > train[1] > train[2] and val[0] < val[1] < val[2]:
        return ("Training loss fell while validation loss rose for three comparable "
                "evaluations; inspect the primary validation metric and supervision counts.")
    return None
