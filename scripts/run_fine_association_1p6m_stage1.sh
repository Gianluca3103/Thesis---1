#!/usr/bin/env bash
# Stage-I-only 1.6 m / 32-candidate ablation.
set -euo pipefail

REPO=${REPO:-/mnt/3D10B36523559581/Gianluca/sensor-fusion-fault-localization}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SOURCE_STUDENT=${SOURCE_STUDENT:-$DATA/crossmodal_full_refog_20261029/crossmodal/full/student/best_radar_iou.ckpt}
CONFIG=${CONFIG:-$REPO/configs/crossmodal_fine_association_1p6m32.json}
OUT=${OUT:-$DATA/fine_association_1p6m32_20261031}
WORKERS=${WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
SWEEP_WORKERS=${SWEEP_WORKERS:-2}

fail() { printf '%s\n' "$*" >&2; exit 1; }
for required in "$PY" "$SOURCE_STUDENT" "$CONFIG" \
                "$RAW/train/manifest.json" "$RAW/val/manifest.json"; do
  [[ -s "$required" ]] || fail "Missing required file: $required"
done
[[ "$WORKERS" =~ ^[0-9]+$ && "$SWEEP_WORKERS" =~ ^[0-9]+$ &&
   "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "Invalid batch or worker count"

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

smoke=$OUT/smoke_stage1
if [[ ! -s "$smoke/last.ckpt" ]]; then
  "$PY" -u -m scripts.train_fine_association_stage1 \
    --vod-root "$VOD" --source-student "$SOURCE_STUDENT" \
    --data-cache-root "$RAW" --output-root "$smoke" --config "$CONFIG" \
    --epochs 1 --train-limit "$BATCH_SIZE" --val-limit 1 \
    --batch-size "$BATCH_SIZE" --num-workers 2 --device cuda
fi
"$PY" - "$smoke/last.ckpt" <<'PY'
import sys
import torch

saved = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
config = saved["config"]
assert saved["radius_m"] == config["offset_limit_m"] == config["target_radius_m"] == 1.6
assert config["proposals_per_anchor"] == 32
print("Smoke checkpoint: 32 radar-only pointers within 1.6 m", flush=True)
PY

stage1=$OUT/stage1
resume=()
if [[ -s "$stage1/last.ckpt" ]]; then
  resume=(--resume "$stage1/last.ckpt")
elif [[ -e "$stage1/metrics.jsonl" ]]; then
  fail "Incomplete Stage-I state in $stage1"
fi
"$PY" -u -m scripts.train_fine_association_stage1 \
  --vod-root "$VOD" --source-student "$SOURCE_STUDENT" \
  --data-cache-root "$RAW" --output-root "$stage1" --config "$CONFIG" \
  --epochs 25 --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" \
  --device cuda "${resume[@]}" 2>&1 | tee -a "$OUT/stage1.log"
[[ -s "$stage1/best.ckpt" ]] || fail "No best 1.6 m Stage-I checkpoint"

sweep=$OUT/sweep_val
"$PY" -u -m scripts.compare_fine_association_edge \
  --vod-root "$VOD" --checkpoint "$stage1/best.ckpt" \
  --split val --limit 1296 --metrics-only \
  --batch-size "$BATCH_SIZE" --num-workers "$SWEEP_WORKERS" \
  --output-root "$sweep" --device cuda 2>&1 | tee "$OUT/sweep.log"

best=$("$PY" - "$sweep/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(summary["best_thresholds"]["radar_1_6m"]["raw"]["f1_at_0_2m"])
PY
)
"$PY" -u -m scripts.compare_fine_association_edge \
  --vod-root "$VOD" --checkpoint "$stage1/best.ckpt" \
  --split val --limit 5 --random-seed 3103 \
  --threshold "$best" --audit-scope radar_1_6m \
  --batch-size 4 --num-workers "$SWEEP_WORKERS" \
  --output-root "$OUT/audit_val5" --device cuda

printf 'Stage I checkpoint: %s\nFull validation sweep: %s\nAudit viewer: %s\n' \
  "$stage1/best.ckpt" "$sweep/summary.json" "$OUT/audit_val5/index.html"
