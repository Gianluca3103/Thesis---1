#!/usr/bin/env bash
set -Eeuo pipefail

# Foreground, from-scratch 30-epoch radar encoder, LiDAR encoder, teacher,
# and student run. Fixed caches are built at the earliest valid phase.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
OUT=${OUT:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/crossmodal_fov70_refined_noaug_30ep_20261015}
RADAR_VARIANT=radar_20frames_verified_doppler_radial
NOAUG=$REPO/configs/crossmodal_no_geometric_augmentation.json
TEACHER_CONFIG=$REPO/configs/crossmodal_teacher_raw64.json
STUDENT_CONFIG=$REPO/configs/crossmodal_student_raw64.json
RAW=$OUT/fixed_frames
TEACHER_CACHE=$OUT/teacher_static
STUDENT_CACHE_FORMAT=${STUDENT_CACHE_FORMAT:-legacy}
STUDENT_COMPRESSION_CODEC=${STUDENT_COMPRESSION_CODEC:-zlib}
STUDENT_COMPRESSION_LEVEL=${STUDENT_COMPRESSION_LEVEL:-6}
STUDENT_CACHE=$OUT/student_supervision
STUDENT_CACHE_ARGS=()
if [[ "$STUDENT_CACHE_FORMAT" == compact ]]; then
  STUDENT_CACHE=$OUT/student_supervision_compact
  STUDENT_CACHE_ARGS=(--cache-format compact
                      --compression-codec "$STUDENT_COMPRESSION_CODEC"
                      --compression-level "$STUDENT_COMPRESSION_LEVEL")
elif [[ "$STUDENT_CACHE_FORMAT" != legacy ]]; then
  printf 'STUDENT_CACHE_FORMAT must be legacy or compact.\n' >&2
  exit 1
fi
SMOKE=$OUT/smoke
SMOKE_STUDENT_CACHE=$SMOKE/student_supervision
if [[ "$STUDENT_CACHE_FORMAT" == compact ]]; then
  SMOKE_STUDENT_CACHE=$SMOKE/student_supervision_compact
fi
FULL=$OUT/full
ENCODER_BATCH_SIZE=${ENCODER_BATCH_SIZE:-16}
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-2}
MIN_FREE_GIB=${MIN_FREE_GIB:-55}
if [[ -s "$RAW/train/manifest.json" || -s "$FULL/radar/last.ckpt" ]]; then
  MIN_FREE_GIB=${RESUME_MIN_FREE_GIB:-10}
fi

on_error() {
  local status=$?
  printf 'Stopped at line %s (exit %s). Outputs remain reusable under %s\n' \
    "$1" "$status" "$OUT" >&2
  exit "$status"
}
trap 'on_error $LINENO' ERR

test -s "$PY"
test -d "$VOD/lidar"
test -d "$VOD/$RADAR_VARIANT"
test -s "$NOAUG"
test -s "$TEACHER_CONFIG"
test -s "$STUDENT_CONFIG"
if (( ENCODER_BATCH_SIZE < 1 || VALIDATION_BATCH_SIZE < 1 )); then
  printf 'Batch sizes must be positive.\n' >&2
  exit 1
fi
mkdir -p "$OUT" "$OUT/matplotlib"
exec 9>"$OUT/run.lock"
flock -n 9 || { printf 'Another run holds %s/run.lock\n' "$OUT" >&2; exit 1; }
FREE_BYTES=$(df -B1 --output=avail "$OUT" | tail -n 1 | tr -d ' ')
if (( FREE_BYTES < MIN_FREE_GIB * 1024 * 1024 * 1024 )); then
  printf 'Need at least %s GiB free for fixed caches on %s\n' "$MIN_FREE_GIB" "$OUT" >&2
  exit 1
fi
SHORT_TMP=${CROSSMODAL_TMPDIR:-/dev/shm/cm-$UID}
mkdir -p "$SHORT_TMP"
export TMPDIR="$SHORT_TMP" TMP="$SHORT_TMP" TEMP="$SHORT_TMP"
export MPLCONFIGDIR="$OUT/matplotlib" MPLBACKEND=Agg
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8
export CROSSMODAL_GPU_NEIGHBORS=${CROSSMODAL_GPU_NEIGHBORS:-1}
cd "$REPO"

"$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["enabled"] is False' "$NOAUG"
"$PY" -c 'import torch; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0), flush=True)'
"$PY" -m scripts.smoke_crossmodal_raw_decoder
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
    printf 'Metrics exist without a usable checkpoint: %s\n' "$root" >&2
    return 1
  else
    printf '%s: starting from epoch 1\n' "$name"
  fi
  "$PY" -u -m "$module" "$@" --output-root "$root" "${resume[@]}" \
    2>&1 | tee -a "$root/console.log"
  test -s "$root/last.ckpt"
}

RAW_ARGS=(--vod-root "$VOD" --cache-root "$RAW"
          --radar-variant "$RADAR_VARIANT" --num-workers 4)
COMMON=(--vod-root "$VOD" --radar-variant "$RADAR_VARIANT"
        --online-augmentation-config "$NOAUG" --data-cache-root "$RAW"
        --device cuda --no-tensorboard)
SMOKE_ARGS=(--epochs 1 --batch-size 1 --train-limit 2 --val-limit 2
            --quick-val-frames 2 --full-val-every 1 --num-workers 0)

printf '\n=== Cache two train and two validation frames for the smoke test ===\n'
"$PY" -u -m scripts.cache_crossmodal_fixed_frames "${RAW_ARGS[@]}" --split train --limit 2
"$PY" -u -m scripts.cache_crossmodal_fixed_frames "${RAW_ARGS[@]}" --split val --limit 2

printf '\n=== Smoke test: independent encoders without augmentation ===\n'
run_phase 'smoke radar encoder' "$SMOKE/radar" \
  models.radar_lidar_crossmodal_v1.pretrain_encoder \
  "${COMMON[@]}" "${SMOKE_ARGS[@]}" --modality radar
run_phase 'smoke LiDAR encoder' "$SMOKE/lidar" \
  models.radar_lidar_crossmodal_v1.pretrain_encoder \
  "${COMMON[@]}" "${SMOKE_ARGS[@]}" --modality lidar
test -s "$SMOKE/radar/best_encoder.pth"
test -s "$SMOKE/lidar/best_encoder.pth"

printf '\n=== Smoke test: cache teacher inputs and train cross-attention ===\n'
for split in train val; do
  "$PY" -u -m scripts.cache_crossmodal_teacher_static \
    --vod-root "$VOD" --data-cache-root "$RAW" --cache-root "$SMOKE/teacher_static" \
    --radar-encoder "$SMOKE/radar/best_encoder.pth" \
    --lidar-encoder "$SMOKE/lidar/best_encoder.pth" --config "$TEACHER_CONFIG" \
    --radar-variant "$RADAR_VARIANT" --split "$split" --limit 2 \
    --num-workers 0 --device cuda
done
run_phase 'smoke teacher' "$SMOKE/teacher" \
  models.radar_lidar_crossmodal_v1.train_teacher \
  "${COMMON[@]}" "${SMOKE_ARGS[@]}" --config "$TEACHER_CONFIG" \
  --static-cache-root "$SMOKE/teacher_static" \
  --radar-encoder "$SMOKE/radar/best_encoder.pth" \
  --lidar-encoder "$SMOKE/lidar/best_encoder.pth"
test -s "$SMOKE/teacher/best_object_r1.ckpt"

printf '\n=== Smoke test: cache student supervision and train radar-only student ===\n'
for split in train val; do
  LIMIT_ARG="--${split}-limit"
  "$PY" -u -m scripts.cache_crossmodal_student_validation \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --teacher-checkpoint "$SMOKE/teacher/best_object_r1.ckpt" \
    --cache-root "$SMOKE_STUDENT_CACHE/$split" \
    "${STUDENT_CACHE_ARGS[@]}" \
    --radar-variant "$RADAR_VARIANT" --split "$split" \
    "$LIMIT_ARG" 2 --num-workers 0 --device cuda
done
run_phase 'smoke student' "$SMOKE/student" \
  models.radar_lidar_crossmodal_v1.train_student \
  "${COMMON[@]}" "${SMOKE_ARGS[@]}" --config "$STUDENT_CONFIG" \
  --teacher-checkpoint "$SMOKE/teacher/best_object_r1.ckpt" \
  --training-cache-root "$SMOKE_STUDENT_CACHE/train" \
  --validation-cache-root "$SMOKE_STUDENT_CACHE/val" \
  --validation-batch-size "$VALIDATION_BATCH_SIZE"
printf 'All cached no-augmentation smoke phases passed.\n'

printf '\n=== Build complete shared fixed frame cache ===\n'
"$PY" -u -m scripts.cache_crossmodal_fixed_frames "${RAW_ARGS[@]}" --split train
"$PY" -u -m scripts.cache_crossmodal_fixed_frames "${RAW_ARGS[@]}" --split val

printf '\n=== Train independent encoders for 30 epochs each ===\n'
"$PY" -u -m scripts.train_crossmodal_encoder_pair \
  --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
  --online-augmentation-config "$NOAUG" --data-cache-root "$RAW" \
  --output-root "$FULL" --epochs 30 --batch-size "$ENCODER_BATCH_SIZE" \
  --full-val-every 5 --num-workers 2 --device cuda
test -s "$FULL/radar/best_encoder.pth"
test -s "$FULL/lidar/best_encoder.pth"

printf '\n=== Build complete frozen teacher input cache ===\n'
for split in train val; do
  "$PY" -u -m scripts.cache_crossmodal_teacher_static \
    --vod-root "$VOD" --data-cache-root "$RAW" --cache-root "$TEACHER_CACHE" \
    --radar-encoder "$FULL/radar/best_encoder.pth" \
    --lidar-encoder "$FULL/lidar/best_encoder.pth" --config "$TEACHER_CONFIG" \
    --radar-variant "$RADAR_VARIANT" --split "$split" \
    --num-workers 2 --device cuda
done

printf '\n=== Train cross-attention teacher for 30 epochs ===\n'
run_phase 'cross-attention teacher' "$FULL/teacher" \
  models.radar_lidar_crossmodal_v1.train_teacher \
  "${COMMON[@]}" --epochs 30 --batch-size 2 --quick-val-frames 64 \
  --full-val-every 5 --num-workers 2 --config "$TEACHER_CONFIG" \
  --static-cache-root "$TEACHER_CACHE" \
  --radar-encoder "$FULL/radar/best_encoder.pth" \
  --lidar-encoder "$FULL/lidar/best_encoder.pth"
test -s "$FULL/teacher/best_object_r1.ckpt"

printf '\n=== Build complete student training and validation caches ===\n'
for split in train val; do
  "$PY" -u -m scripts.cache_crossmodal_student_validation \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --teacher-checkpoint "$FULL/teacher/best_object_r1.ckpt" \
    --cache-root "$STUDENT_CACHE/$split" \
    "${STUDENT_CACHE_ARGS[@]}" \
    --radar-variant "$RADAR_VARIANT" --split "$split" \
    --num-workers 2 --device cuda
done

printf '\n=== Train radar-only confidence student for 30 epochs ===\n'
run_phase 'radar-only student' "$FULL/student" \
  models.radar_lidar_crossmodal_v1.train_student \
  "${COMMON[@]}" --epochs 30 --batch-size 1 --quick-val-frames 32 \
  --full-val-every 5 --num-workers 2 --config "$STUDENT_CONFIG" \
  --teacher-checkpoint "$FULL/teacher/best_object_r1.ckpt" \
  --training-cache-root "$STUDENT_CACHE/train" \
  --validation-cache-root "$STUDENT_CACHE/val" \
  --validation-batch-size "$VALIDATION_BATCH_SIZE"
printf '\nAll four 30-epoch runs completed under %s\n' "$FULL"
