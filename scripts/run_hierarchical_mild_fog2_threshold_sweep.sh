#!/usr/bin/env bash
# Matched 20%-validation threshold sweeps for old/new Stage I and three target scopes.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-hierarchical-stage1-20261101}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
NEW=${NEW:-$DATA/hierarchical_stage1_mild_fog2_20261101/best.ckpt}
OLD=${OLD:-$DATA/hierarchical_stage1_expanded_full_20261101/best.ckpt}
OUT=${OUT:-$DATA/hierarchical_stage1_mild_fog2_20261101/threshold_sweep_val20}

for required in "$PY" "$NEW" "$OLD" "$MILD/val/manifest.json" \
  "$REPO/scripts/compare_hierarchical_stage1.py"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for scope in full_roi radar3_1m radar5_1m; do
  scope_args=()
  if [[ "$scope" == radar3_1m ]]; then
    scope_args=(--radar-radius-m 1 --radar-min-points 3)
  elif [[ "$scope" == radar5_1m ]]; then
    scope_args=(--radar-radius-m 1 --radar-min-points 5)
  fi
  for label in new old; do
    checkpoint=$NEW
    if [[ "$label" == old ]]; then checkpoint=$OLD; fi
    "$PY" -u -m scripts.compare_hierarchical_stage1 \
      --vod-root "$VOD" --checkpoint "$checkpoint" \
      --faulty-root "$MILD" --evaluate-modes mild_dataset \
      --output "$OUT/${scope}_${label}.json" \
      --limit 512 --seed 3103 --temporal-gap 0 \
      --budgets 250 500 1000 2000 5000 10000 15000 20000 30000 \
      --minimum-precision 0.20 --threshold-steps 10000 \
      --device cuda "${scope_args[@]}" \
      > "$OUT/${scope}_${label}.log" 2>&1
    printf '%s %s complete: %s\n' "$scope" "$label" "$OUT/${scope}_${label}.json"
  done
done

"$PY" - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
labels = {
    "best_recall_at_minimum_precision": "max recall with precision >=20%",
    "precision_at_90_percent_recall": "max precision with recall >=90%",
    "precision_at_95_percent_recall": "max precision with recall >=95%",
}
reports = {}
for scope in ("full_roi", "radar3_1m", "radar5_1m"):
    for model in ("new", "old"):
        reports[scope, model] = json.loads((root / f"{scope}_{model}.json").read_text())
reference = reports["full_roi", "new"]["sampled_frame_ids"]
if any(report["sampled_frame_ids"] != reference for report in reports.values()):
    raise SystemExit("The six sweeps sampled different validation frames; do not compare them.")
if len(reference) != 512:
    raise SystemExit(f"Expected 512 sampled frames, got {len(reference)}")
print(f"Matched validation frames: {len(reference)} = 256 calibration + 256 held-out")
for scope in ("full_roi", "radar3_1m", "radar5_1m"):
    print(f"\n{scope}")
    for model in ("old", "new"):
        report = reports[scope, model]
        pool = report["heldout"]["hierarchical"]["mild_dataset"]["pool"]
        print(f"  {model}: target={pool['target']:,}, pool-oracle recall={pool['recall']:.2%}")
        for key, title in labels.items():
            descriptive = report["heldout_operating_points"]["mild_dataset"][key]
            calibrated = report["threshold_selection"]["mild_dataset"][key]
            if descriptive is None:
                print(f"    {title}: unavailable on the held-out threshold grid")
            else:
                average_cells = descriptive["selected"] / report["heldout_frames"]
                print(f"    {title}: P={descriptive['precision']:.2%}, "
                      f"R={descriptive['recall']:.2%}, threshold={descriptive['threshold']:.4f}, "
                      f"mean cells/frame={average_cells:,.0f} [descriptive held-out optimum]")
            if calibrated is None:
                print("      calibration-selected threshold: unavailable")
            else:
                row = calibrated["heldout"]
                average_cells = row["selected"] / report["heldout_frames"]
                print(f"      calibration-selected threshold {row['threshold']:.4f} "
                      f"on held-out: P={row['precision']:.2%}, R={row['recall']:.2%}, "
                      f"mean cells/frame={average_cells:,.0f}")
print(f"\nDetailed reports: {root}")
PY
