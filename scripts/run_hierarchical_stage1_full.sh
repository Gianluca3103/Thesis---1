#!/usr/bin/env bash
# Full coarse-to-fine Stage-I coordinate trial and matched old-gate comparison.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-hierarchical-stage1-20261101}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
CONFIG=${CONFIG:-$REPO/configs/hierarchical_stage1_gate_expanded_v1.json}
OUT=${OUT:-$DATA/hierarchical_stage1_expanded_full_20261101}
BASELINE=${BASELINE:-$DATA/fine_association_1p6m64_20261031/stage1/best.ckpt}
WORKERS=${WORKERS:-4}

fail() { printf '%s\n' "$*" >&2; exit 1; }
[[ "$WORKERS" =~ ^[0-9]+$ ]] || fail "WORKERS must be a nonnegative integer"
for required in "$PY" "$CONFIG" "$REPO/scripts/train_hierarchical_stage1.py" \
                "$REPO/scripts/compare_hierarchical_stage1.py" \
                "$REPO/models/radar_lidar_crossmodal_v1/hierarchical_gate.py" \
                "$RAW/train/manifest.json" "$RAW/val/manifest.json"; do
  [[ -s "$required" ]] || fail "Missing required file: $required"
done
[[ -d "$VOD" ]] || fail "Missing VoD directory: $VOD"

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PY" - <<'PY'
from pathlib import Path
import torch
import scipy
from models.radar_lidar_crossmodal_v1 import hierarchical_gate
assert torch.cuda.is_available(), "CUDA is required for the full Stage-I run"
expected = Path.cwd() / "models/radar_lidar_crossmodal_v1/hierarchical_gate.py"
assert Path(hierarchical_gate.__file__).resolve() == expected.resolve(), (
    f"Imported stale Stage-I code from {hierarchical_gate.__file__}")
print(f"Torch {torch.__version__}; CUDA {torch.cuda.get_device_name(0)}; SciPy {scipy.__version__}", flush=True)
PY

if [[ ! -s "$OUT/smoke/best.ckpt" ]]; then
  "$PY" -u -m scripts.train_hierarchical_stage1 \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --output-root "$OUT/smoke" --config "$CONFIG" \
    --train-limit 1 --val-limit 2 --val-selection blocks \
    --epochs 1 --num-workers 0 --budgets 10000 --device cuda \
    2>&1 | tee "$OUT/smoke.log"
fi
[[ -s "$OUT/smoke/best.ckpt" ]] || fail "Smoke checkpoint was not created"

resume=()
if [[ -s "$OUT/last.ckpt" ]]; then
  resume=(--resume "$OUT/last.ckpt")
elif [[ -e "$OUT/metrics.jsonl" ]]; then
  fail "Metrics exist without a resumable checkpoint in $OUT"
fi

"$PY" -u -m scripts.train_hierarchical_stage1 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --output-root "$OUT" --config "$CONFIG" \
  --epochs 25 --val-limit 128 --val-selection blocks \
  --num-workers "$WORKERS" --seed 3103 \
  --budgets 5000 10000 15000 20000 30000 \
  --device cuda "${resume[@]}" 2>&1 | tee -a "$OUT/train.log"
[[ -s "$OUT/best.ckpt" ]] || fail "Full-run checkpoint was not created"

comparison=(
  --vod-root "$VOD" --checkpoint "$OUT/best.ckpt"
  --limit 256 --temporal-gap 20
  --budgets 5000 10000 15000 20000 30000
  --output "$OUT/matched_heldout.json" --device cuda
)
if [[ -s "$BASELINE" ]]; then
  comparison+=(--baseline-checkpoint "$BASELINE")
else
  printf 'Old 64-slot checkpoint absent; writing new-gate-only comparison. Expected: %s\n' "$BASELINE" >&2
fi
"$PY" -u -m scripts.compare_hierarchical_stage1 "${comparison[@]}" \
  2>&1 | tee "$OUT/matched_heldout.log"

printf 'New Stage-I checkpoint: %s\nComparison: %s\n' \
  "$OUT/best.ckpt" "$OUT/matched_heldout.json"
