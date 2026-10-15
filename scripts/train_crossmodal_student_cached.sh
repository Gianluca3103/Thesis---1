#!/usr/bin/env bash
set -Eeuo pipefail

# Foreground validation-cache build, two-frame integration check, then resume
# or start the radar-only student. Existing encoder/teacher checkpoints stay fixed.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
OUT=${OUT:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/crossmodal_native_sparse_verified_20261015_25ep}
TEACHER=$OUT/full/teacher/best_object_r1.ckpt
RUN=$OUT/full/student
CACHE=$OUT/student_validation_cache
RADAR_VARIANT=radar_20frames_verified_doppler_radial
AUGMENTATION=$REPO/configs/stage1_online_geometric_augmentation.json
CONFIG=$REPO/configs/crossmodal_student_raw64.json

test -s "$PY"
test -s "$TEACHER"
test -s "$AUGMENTATION"
test -s "$CONFIG"
mkdir -p "$OUT" "$RUN" "$OUT/matplotlib"
exec 9>"$OUT/run.lock"
flock -n 9 || { printf 'Another cross-modal run holds %s/run.lock\n' "$OUT" >&2; exit 1; }
SHORT_TMP=${CROSSMODAL_TMPDIR:-/dev/shm/cm-$UID}
mkdir -p "$SHORT_TMP"
export TMPDIR="$SHORT_TMP" TMP="$SHORT_TMP" TEMP="$SHORT_TMP"
export MPLCONFIGDIR="$OUT/matplotlib" MPLBACKEND=Agg
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8
export CROSSMODAL_GPU_NEIGHBORS=${CROSSMODAL_GPU_NEIGHBORS:-1}
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-2}
if (( VALIDATION_BATCH_SIZE < 1 )); then
  printf 'VALIDATION_BATCH_SIZE must be at least 1.\n' >&2
  exit 1
fi
cd "$REPO"

CACHE_ARGS=(--vod-root "$VOD" --teacher-checkpoint "$TEACHER"
            --cache-root "$CACHE" --radar-variant "$RADAR_VARIANT"
            --num-workers 2 --device cuda)
if [[ -s "$RUN/last.ckpt" ]]; then
  CACHE_ARGS+=(--student-checkpoint "$RUN/last.ckpt")
fi

printf '\n=== Build two-frame cache and run a foreground integration check ===\n'
"$PY" -u -m scripts.cache_crossmodal_student_validation \
  "${CACHE_ARGS[@]}" --val-limit 2
SMOKE="$OUT/smoke/student_cached_$(git rev-parse --short HEAD)"
if [[ ! -s "$SMOKE/last.ckpt" ]]; then
  "$PY" -u -m models.radar_lidar_crossmodal_v1.train_student \
    --vod-root "$VOD" --teacher-checkpoint "$TEACHER" \
    --radar-variant "$RADAR_VARIANT" --output-root "$SMOKE" --config "$CONFIG" \
    --online-augmentation-config "$AUGMENTATION" \
    --validation-cache-root "$CACHE" --validation-batch-size "$VALIDATION_BATCH_SIZE" \
    --epochs 1 --batch-size 1 --train-limit 1 --val-limit 2 \
    --quick-val-frames 2 --full-val-every 1 --num-workers 0 \
    --device cuda --no-tensorboard
fi
test -s "$SMOKE/last.ckpt"

printf '\n=== Build full fixed validation cache ===\n'
"$PY" -u -m scripts.cache_crossmodal_student_validation "${CACHE_ARGS[@]}"

printf '\n=== Train radar-only student with cached validation ===\n'
RESUME=()
if [[ -s "$RUN/last.ckpt" ]]; then
  RESUME=(--resume "$RUN/last.ckpt")
  printf 'Resuming student from %s\n' "$RUN/last.ckpt"
elif [[ -e "$RUN/metrics.jsonl" ]]; then
  printf 'Student metrics exist without last.ckpt; preserving run and stopping.\n' >&2
  exit 1
fi
"$PY" -u -m models.radar_lidar_crossmodal_v1.train_student \
  --vod-root "$VOD" --teacher-checkpoint "$TEACHER" \
  --radar-variant "$RADAR_VARIANT" --output-root "$RUN" --config "$CONFIG" \
  --online-augmentation-config "$AUGMENTATION" \
  --validation-cache-root "$CACHE" --validation-batch-size "$VALIDATION_BATCH_SIZE" \
  --epochs 25 --batch-size 1 --quick-val-frames 32 --full-val-every 5 \
  --num-workers 2 --device cuda --no-tensorboard "${RESUME[@]}" \
  2>&1 | tee -a "$RUN/console.log"
