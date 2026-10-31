#!/usr/bin/env bash
# Train 64 radar pointers at 1.6 m, then evaluate the chosen top-K view.
set -euo pipefail

REPO=${REPO:-/mnt/3D10B36523559581/Gianluca/sensor-fusion-fault-localization}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SOURCE_STUDENT=${SOURCE_STUDENT:-$DATA/crossmodal_full_refog_20261029/crossmodal/full/student/best_radar_iou.ckpt}
CONFIG=${CONFIG:-$REPO/configs/crossmodal_fine_association_1p6m64.json}
OUT=${OUT:-$DATA/fine_association_1p6m64_20261031}
WORKERS=${WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
SWEEP_WORKERS=${SWEEP_WORKERS:-2}
TOP_K=${TOP_K:-64}

fail() { printf '%s\n' "$*" >&2; exit 1; }
for required in "$PY" "$SOURCE_STUDENT" "$CONFIG" \
                "$RAW/train/manifest.json" "$RAW/val/manifest.json"; do
  [[ -s "$required" ]] || fail "Missing required file: $required"
done
[[ "$WORKERS" =~ ^[0-9]+$ && "$SWEEP_WORKERS" =~ ^[0-9]+$ &&
   "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "Invalid batch or worker count"
case "$TOP_K" in 32|48|64) ;; *) fail "TOP_K must be 32, 48, or 64" ;; esac

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
"$PY" - "$smoke/last.ckpt" "$VOD" "$RAW" <<'PY'
import sys
import torch
from models.radar_lidar_crossmodal_v1.fine_association import load_fine_association
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset

saved = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
config = saved["config"]
assert saved["radius_m"] == config["offset_limit_m"] == config["target_radius_m"] == 1.6
assert config["proposals_per_anchor"] == config["target_neighbors"] == 64
model, _ = load_fine_association(sys.argv[1], "cuda")
frame = paired_dataset(sys.argv[2], "val", saved["radar_variant"],
                       data_cache_root=sys.argv[3], limit=1)[0]
radar = frame["radar"].cuda()
valid = torch.ones((1, len(radar)), dtype=torch.bool, device="cuda")
with torch.no_grad():
    output = model(radar[None], valid)
    for count in (32, 48, 64):
        filtered = output.exported(top_k_per_anchor=count)
        assert filtered["surface_xyz"].shape[1] == count
        assert filtered["interface_version"] == "crossmodal_fine_association_v1"
print("Smoke checkpoint: 64 radar-only pointers within 1.6 m", flush=True)
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
[[ -s "$stage1/best.ckpt" ]] || fail "No best 1.6 m / 64-slot checkpoint"

sweep=$OUT/sweep_val_top${TOP_K}
"$PY" -u -m scripts.compare_fine_association_edge \
  --vod-root "$VOD" --checkpoint "$stage1/best.ckpt" \
  --split val --limit 1296 --metrics-only --top-k "$TOP_K" \
  --batch-size "$BATCH_SIZE" --num-workers "$SWEEP_WORKERS" \
  --output-root "$sweep" --device cuda 2>&1 | tee "$OUT/sweep_top${TOP_K}.log"

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
  --threshold "$best" --audit-scope radar_1_6m --top-k "$TOP_K" \
  --batch-size 4 --num-workers "$SWEEP_WORKERS" \
  --output-root "$OUT/audit_val5_top${TOP_K}" --device cuda

printf 'Stage I checkpoint: %s\nFull validation sweep: %s\nAudit viewer: %s\n' \
  "$stage1/best.ckpt" "$sweep/summary.json" "$OUT/audit_val5_top${TOP_K}/index.html"
