"""Print matched validation results from the old radar AE and D-PCC adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--dpcc", type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.dpcc.read_text(encoding="utf-8"))
    if old["split"] != "val" or new["split"] != "val":
        parser.error("Both reports must be validation results")
    if old["frames"] != new["frames"]:
        parser.error("Validation frame counts differ; comparison is invalid")
    print(f"Matched filtered radar validation: {old['frames']} frames")
    print("Model | selected by | threshold | voxel IoU | point F1@0.2m | "
          "point P/R@0.2m | Chamfer m2")
    for name, report in (("Previous radar AE", old), ("Radar D-PCC", new)):
        for selector, key in (("point F1", "point_f1_0p2m"),
                              ("voxel IoU", "voxel_iou")):
            row = max(report["rows"], key=lambda item: item[key])
            chamfer = row.get("chamfer_squared_m2")
            chamfer_text = f"{chamfer:.4f}" if chamfer is not None else "unavailable"
            print(f"{name} | {selector} | {row['threshold']:.2f} | "
                  f"{row['voxel_iou']:.2%} | {row['point_f1_0p2m']:.2%} | "
                  f"{row['point_precision_0p2m']:.2%}/"
                  f"{row['point_recall_0p2m']:.2%} | {chamfer_text}")


if __name__ == "__main__":
    main()
