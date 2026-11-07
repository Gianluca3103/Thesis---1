"""Print the matched clean Stage-II checkpoint-by-sampler comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for model in ("old", "new", "new_iou"):
        for strategy in ("one_step", "reverse3"):
            prefix = f"{model}_{strategy}"
            val = json.loads((args.report_root / f"{prefix}_val.json").read_text())
            test = json.loads((args.report_root / f"{prefix}_test.json").read_text())
            if (val["checkpoint"] != test["checkpoint"] or
                    val["occupancy_threshold"] != test["occupancy_threshold"] or
                    val["sampling"] != test["sampling"] or
                    val["metrics"]["all_validation"]["frames"] != 1296 or
                    test["metrics"]["test"]["frames"] != 2247):
                raise ValueError(f"Inconsistent or incomplete reports for {prefix}")
            rows.append({"model": model, "sampling": strategy,
                         "checkpoint": val["checkpoint"],
                         "checkpoint_epoch": val["checkpoint_epoch"],
                         "threshold": val["occupancy_threshold"],
                         "validation": val["metrics"]["all_validation"],
                         "test": test["metrics"]["test"]})
    output = args.report_root / "comparison_summary.json"
    output.write_text(json.dumps({"rows": rows}, indent=2) + "\n",
                      encoding="utf-8")
    print("Model  Sampler  Epoch  Threshold  Val IoU  Test IoU  "
          "Val IoU@0.2m  Test IoU@0.2m  Test F1  Test Chamfer m²")
    for row in rows:
        val, test = row["validation"], row["test"]
        chamfer = test["chamfer_squared_l2_m2_paired_frame_mean"]
        print(f"{row['model']:5}  {row['sampling']:8}  "
              f"{row['checkpoint_epoch']:5}  {row['threshold']:9.3f}  "
              f"{val['iou']:7.2%}  {test['iou']:8.2%}  "
              f"{val['iou_0p2m']:12.2%}  {test['iou_0p2m']:13.2%}  "
              f"{test['f1']:7.2%}  "
              f"{chamfer if chamfer is not None else float('nan'):.6f}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
