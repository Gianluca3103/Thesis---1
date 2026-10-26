"""Select the saved SVEFusion epoch with the highest clean-validation mAP."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


SCORE = re.compile(r"Validation epoch (\d+): 3D AP_R40 moderate .*?mAP=([\d.]+)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, required=True)
    args = parser.parse_args()
    output = args.training_output
    candidates = []
    for log in sorted(output.glob("train_*.log")):
        for match in SCORE.finditer(log.read_text(encoding="utf-8", errors="replace")):
            epoch, score = int(match.group(1)), float(match.group(2))
            checkpoint = output / "ckpt" / f"checkpoint_epoch_{epoch}.pth"
            if checkpoint.is_file() and checkpoint.stat().st_size:
                candidates.append((score, epoch, checkpoint))
    if not candidates:
        raise FileNotFoundError(f"No scored SVEFusion checkpoint under {output}")
    score, epoch, path = max(candidates)
    print(f"Selected clean-validation epoch {epoch}, moderate mAP_R40={score:.2f}",
          file=sys.stderr, flush=True)
    print(path)


if __name__ == "__main__":
    main()
