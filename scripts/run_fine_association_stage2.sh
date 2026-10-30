#!/usr/bin/env bash
set -Eeuo pipefail

# New Stage-I 3 m / 32-pointer trial and its own Stage-II gate/diffusion.
# Reuses only the aligned raw frames, trained full-region LPCR, and fault cache.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
SOURCE_RUN=${SOURCE_RUN:-$DATA/crossmodal_full_refog_20261029}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SOURCE_STUDENT=${SOURCE_STUDENT:-$SOURCE_RUN/crossmodal/full/student/best_radar_iou.ckpt}
LPCR_CKPT=${LPCR_CKPT:-$SOURCE_RUN/lpcr/best.ckpt}
SOURCE_STAGE2=${SOURCE_STAGE2:-$SOURCE_RUN/stage2/diffusion/best.ckpt}
OUT=${OUT:-$DATA/fine_association_3m32_20261030}
CONFIG=${CONFIG:-$REPO/configs/crossmodal_fine_association_3m32.json}
INITIAL_ASSOCIATION=${INITIAL_ASSOCIATION-$DATA/fine_association_3m8_20261030/stage1/best.ckpt}
WORKERS=${WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
DIFF_BATCH_SIZE=${DIFF_BATCH_SIZE:-16}
GATE=${GATE:-0.25}

fail() { printf '%s\n' "$*" >&2; exit 1; }
[[ -x "$PY" && -s "$SOURCE_STUDENT" && -s "$LPCR_CKPT" &&
   -s "$RAW/train/manifest.json" && -s "$RAW/val/manifest.json" ]] ||
  fail "Python, trained Stage I, LPCR, or aligned-frame cache is missing"
[[ "$WORKERS" =~ ^[0-9]+$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ &&
   "$DIFF_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "Invalid worker or batch count"

mkdir -p "$OUT" "$OUT/matplotlib"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPLCONFIGDIR="$OUT/matplotlib" MPLBACKEND=Agg
cd "$REPO"
"$PY" -m scripts.smoke_stage2_sparse --device cuda

if [[ -z ${FOG_CACHE:-} ]]; then
  [[ -s "$SOURCE_STAGE2" ]] || fail "Set FOG_CACHE or supply SOURCE_STAGE2"
  FOG_CACHE=$("$PY" -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False).get("fault_cache_root") or "")' "$SOURCE_STAGE2")
fi
[[ -s "$FOG_CACHE/train/manifest.json" && -s "$FOG_CACHE/val/manifest.json" ]] ||
  fail "Revised train/val fog cache missing: $FOG_CACHE"

VARIANT=$("$PY" -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["radar_variant"])' "$SOURCE_STUDENT")
printf 'Radar: %s\nSource student: %s\nFog cache: %s\nOutput: %s\n' \
  "$VARIANT" "$SOURCE_STUDENT" "$FOG_CACHE" "$OUT"
warm_start=()
if [[ -n "$INITIAL_ASSOCIATION" && -s "$INITIAL_ASSOCIATION" ]]; then
  warm_start=(--initialize-from-association "$INITIAL_ASSOCIATION")
  printf 'Warm start: %s\n' "$INITIAL_ASSOCIATION"
fi

smoke=$OUT/smoke_stage1
if [[ ! -s "$smoke/last.ckpt" ]]; then
  "$PY" -u -m scripts.train_fine_association_stage1 \
    --vod-root "$VOD" --source-student "$SOURCE_STUDENT" \
    --data-cache-root "$RAW" --output-root "$smoke" \
    --config "$CONFIG" --epochs 1 --train-limit "$BATCH_SIZE" \
    --val-limit 1 --batch-size "$BATCH_SIZE" --num-workers 2 \
    --device cuda "${warm_start[@]}"
fi
"$PY" - "$smoke/last.ckpt" "$VOD" "$RAW" "$VARIANT" <<'PY'
import sys
from pathlib import Path
import torch
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.stage2_io import load_stage1, stage1_frame

checkpoint, vod, raw = map(Path, sys.argv[1:4])
variant = sys.argv[4]
model, saved = load_stage1(checkpoint, "cuda", variant)
frame = paired_dataset(vod, "val", variant, data_cache_root=raw, limit=1)[0]
radar = frame["radar"].cuda()
valid = torch.ones((1, len(radar)), dtype=torch.bool, device="cuda")
with torch.no_grad():
    proposals = model(radar[None], valid).surface
    output = stage1_frame(model, radar, saved["radius_m"])
distance = torch.linalg.vector_norm(
    proposals.xyz - proposals.anchor_xyz[:, None], dim=-1)
assert proposals.xyz.shape[1] == 32
assert not len(distance) or bool((distance <= 3.0 + 1e-5).all())
assert len(output["confidence_coords"]) == len(output["confidence_probabilities"])
assert output["interface_version"] == "crossmodal_fine_association_v1"
print(f"Fine Stage-I interface OK: {len(proposals.anchor_xyz)} radar voxels, "
      f"{len(output['confidence_coords'])} candidate LiDAR voxels", flush=True)
PY

stage1=$OUT/stage1
resume=()
if [[ -s "$stage1/last.ckpt" ]]; then
  resume=(--resume "$stage1/last.ckpt")
elif [[ -e "$stage1/metrics.jsonl" ]]; then
  fail "Incomplete Stage-I association state in $stage1"
fi
"$PY" -u -m scripts.train_fine_association_stage1 \
  --vod-root "$VOD" --source-student "$SOURCE_STUDENT" \
  --data-cache-root "$RAW" --output-root "$stage1" --config "$CONFIG" \
  --epochs 25 --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" \
  --device cuda "${warm_start[@]}" "${resume[@]}" 2>&1 | tee -a "$OUT/stage1.log"
[[ -s "$stage1/best.ckpt" ]] || fail "No best fine-association checkpoint"

gate_cache=$OUT/stage2_gate
"$PY" -u -m scripts.cache_stage2_sparse \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --stage1-checkpoint "$stage1/best.ckpt" \
  --cache-root "$OUT/stage2_gate_smoke" \
  --radar-variant "$VARIANT" --gate-threshold "$GATE" \
  --split train --limit 2 --num-workers 2 --device cuda
"$PY" -u -m scripts.cache_stage2_sparse \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --stage1-checkpoint "$stage1/best.ckpt" --cache-root "$gate_cache" \
  --radar-variant "$VARIANT" --gate-threshold "$GATE" \
  --split both --num-workers "$WORKERS" --device cuda

diff=$OUT/diffusion
resume=()
if [[ -s "$diff/last.ckpt" ]]; then
  resume=(--resume "$diff/last.ckpt")
elif [[ -e "$diff/metrics.jsonl" ]]; then
  fail "Incomplete Stage-II diffusion state in $diff"
fi
"$PY" -u -m scripts.train_stage2_sparse \
  --phase diffusion --vod-root "$VOD" --data-cache-root "$RAW" \
  --stage2-cache-root "$gate_cache" --fault-cache-root "$FOG_CACHE" \
  --stage1-checkpoint "$stage1/best.ckpt" \
  --lpcr-checkpoint "$LPCR_CKPT" --output-root "$diff" \
  --radar-variant "$VARIANT" --gate-threshold "$GATE" \
  --epochs 50 --lr 0.0002 --batch-size "$DIFF_BATCH_SIZE" \
  --num-workers "$WORKERS" --residual-weight 1 \
  --reconstruction-weight 1 --addition-weight 1 --healthy-weight 1 \
  --ray-hit-weight 0.25 --ray-extra-weight 0.25 \
  --device cuda "${resume[@]}" 2>&1 | tee -a "$OUT/stage2.log"

printf 'Stage I: %s\nStage II: %s\n' "$stage1/best.ckpt" "$diff/best.ckpt"
