#!/usr/bin/env bash
set -Eeuo pipefail

# Run all three training phases with a one-frame integration check first.
# OUT may be overridden when more space is available on another volume.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
OUT=${OUT:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/crossmodal_native_sparse_verified_20261015_25ep}
RADAR_VARIANT=radar_20frames_verified_doppler_radial
AUGMENTATION=$REPO/configs/stage1_online_geometric_augmentation.json
TEACHER_CONFIG=$REPO/configs/crossmodal_teacher_raw64.json
STUDENT_CONFIG=$REPO/configs/crossmodal_student_raw64.json

on_error() {
  local status=$?
  printf 'Stopped at line %s (exit %s). Inspect %s/overnight.log and the phase console.log.\n' \
    "$1" "$status" "$OUT" >&2
  exit "$status"
}
trap 'on_error $LINENO' ERR

test -s "$PY"
test -d "$VOD/lidar"
test -d "$VOD/$RADAR_VARIANT"
test -s "$AUGMENTATION"
test -s "$TEACHER_CONFIG"
test -s "$STUDENT_CONFIG"
mkdir -p "$OUT"
FREE_BYTES=$(df -B1 --output=avail "$OUT" | tail -n 1 | tr -d ' ')
if (( FREE_BYTES < 15 * 1024 * 1024 * 1024 )); then
  printf 'Need at least 15 GiB free on the output volume: %s\n' "$OUT" >&2
  exit 1
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8
export MPLBACKEND=Agg
cd "$REPO"

"$PY" - <<'PY'
from pathlib import Path
from models.radar_lidar_crossmodal_v1 import INTERFACE_VERSION
from models.radar_lidar_crossmodal_v1 import previous_encoder
from models.radar_lidar_crossmodal_v1.student import StudentConfig
assert INTERFACE_VERSION == "radar_lidar_crossmodal_v4_stage1_native_sparse_raw64"
assert StudentConfig().lambda_teacher_evidence == 0.5
assert hasattr(previous_encoder, "PreviousStage1Sparse")
print("Verified native sparse source:", Path(previous_encoder.__file__).resolve(), flush=True)
PY
"$PY" -c 'import torch; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0), flush=True)'
printf 'Source: %s\nOutput: %s\n' "$(git rev-parse HEAD)" "$OUT"

run_phase() {
  local name=$1 root=$2 module=$3
  shift 3
  mkdir -p "$root"
  local -a resume=()
  if [[ -s "$root/last.ckpt" ]]; then
    resume=(--resume "$root/last.ckpt")
    printf '%s: resuming from %s\n' "$name" "$root/last.ckpt"
  elif [[ -e "$root/metrics.jsonl" ]]; then
    printf '%s has metrics but no usable last.ckpt; preserving it and stopping.\n' "$root" >&2
    return 1
  else
    printf '%s: starting from epoch 1\n' "$name"
  fi
  "$PY" -u -m "$module" "$@" --output-root "$root" "${resume[@]}" \
    2>&1 | tee -a "$root/console.log"
  test -s "$root/last.ckpt"
}

COMMON=(--vod-root "$VOD" --radar-variant "$RADAR_VARIANT"
        --online-augmentation-config "$AUGMENTATION" --device cuda
        --no-tensorboard)
SMOKE_ARGS=(--epochs 1 --batch-size 1 --train-limit 1 --val-limit 1
            --quick-val-frames 1 --full-val-every 1 --num-workers 0)
FULL_ARGS=(--epochs 25 --full-val-every 5 --num-workers 2)
SMOKE=$OUT/smoke
FULL=$OUT/full

printf '\n=== GPU sparse-decoder smoke test ===\n'
"$PY" -m scripts.smoke_crossmodal_raw_decoder 2>&1 | tee -a "$OUT/decoder_smoke.log"

printf '\n=== One-frame end-to-end smoke tests ===\n'
run_phase 'smoke radar encoder' "$SMOKE/radar" \
  models.radar_lidar_crossmodal_v1.pretrain_encoder \
  "${COMMON[@]}" "${SMOKE_ARGS[@]}" --modality radar
test -s "$SMOKE/radar/best_encoder.pth"
run_phase 'smoke LiDAR encoder' "$SMOKE/lidar" \
  models.radar_lidar_crossmodal_v1.pretrain_encoder \
  "${COMMON[@]}" "${SMOKE_ARGS[@]}" --modality lidar
test -s "$SMOKE/lidar/best_encoder.pth"
run_phase 'smoke cross-attention teacher' "$SMOKE/teacher" \
  models.radar_lidar_crossmodal_v1.train_teacher \
  "${COMMON[@]}" "${SMOKE_ARGS[@]}" --config "$TEACHER_CONFIG" \
  --radar-encoder "$SMOKE/radar/best_encoder.pth" \
  --lidar-encoder "$SMOKE/lidar/best_encoder.pth"
SMOKE_TEACHER=$SMOKE/teacher/best_object_r1.ckpt
if [[ ! -s "$SMOKE_TEACHER" ]]; then
  SMOKE_TEACHER=$SMOKE/teacher/last.ckpt
  printf 'Smoke frame had no selected object-R@1 checkpoint; using last.ckpt for the integration check.\n'
fi
run_phase 'smoke radar-only student' "$SMOKE/student" \
  models.radar_lidar_crossmodal_v1.train_student \
  "${COMMON[@]}" "${SMOKE_ARGS[@]}" --config "$STUDENT_CONFIG" \
  --teacher-checkpoint "$SMOKE_TEACHER"
printf 'All four one-frame smoke tests passed. Starting full training.\n'

printf '\n=== Full training: independent encoders ===\n'
run_phase 'radar encoder' "$FULL/radar" \
  models.radar_lidar_crossmodal_v1.pretrain_encoder \
  "${COMMON[@]}" "${FULL_ARGS[@]}" --batch-size 2 --modality radar
test -s "$FULL/radar/best_encoder.pth"
run_phase 'LiDAR encoder' "$FULL/lidar" \
  models.radar_lidar_crossmodal_v1.pretrain_encoder \
  "${COMMON[@]}" "${FULL_ARGS[@]}" --batch-size 2 --modality lidar
test -s "$FULL/lidar/best_encoder.pth"

printf '\n=== Full training: cross-attention teacher ===\n'
run_phase 'cross-attention teacher' "$FULL/teacher" \
  models.radar_lidar_crossmodal_v1.train_teacher \
  "${COMMON[@]}" "${FULL_ARGS[@]}" --batch-size 2 --config "$TEACHER_CONFIG" \
  --radar-encoder "$FULL/radar/best_encoder.pth" \
  --lidar-encoder "$FULL/lidar/best_encoder.pth"
test -s "$FULL/teacher/best_object_r1.ckpt" || {
  printf 'Teacher did not produce best_object_r1.ckpt; student training was not started.\n' >&2
  exit 1
}

printf '\n=== Full training: radar-only confidence student ===\n'
run_phase 'radar-only student' "$FULL/student" \
  models.radar_lidar_crossmodal_v1.train_student \
  "${COMMON[@]}" "${FULL_ARGS[@]}" --batch-size 1 --config "$STUDENT_CONFIG" \
  --teacher-checkpoint "$FULL/teacher/best_object_r1.ckpt"

printf '\nTraining complete. Full-run checkpoints are under %s\n' "$FULL"
if [[ ! -s "$FULL/student/best_radar_iou.ckpt" ]]; then
  printf 'Student finished, but no checkpoint met the known-space IoU coverage gate. Inspect its validation metrics.\n' >&2
fi
