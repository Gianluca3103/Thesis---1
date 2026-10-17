#!/usr/bin/env bash
# Foreground, matched head-only and joint fine-tuning from one pointer checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

: "${VOD:?Set VOD to the View-of-Delft root}"
: "${SURFACE:?Set SURFACE to best_proposal_utility.ckpt}"
: "${OUT:?Set OUT to a new ablation output directory}"

PY=${PY:-python}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
FULL_VAL_EVERY=${FULL_VAL_EVERY:-5}
QUICK_VAL_FRAMES=${QUICK_VAL_FRAMES:-32}
SEED=${SEED:-42}
DISTILLATION_WEIGHT=${DISTILLATION_WEIGHT:-1}
HEAD_LEARNING_RATE=${HEAD_LEARNING_RATE:-5e-5}
FEATURE_LEARNING_RATE=${FEATURE_LEARNING_RATE:-2e-5}
if [[ -n ${TEACHER_CACHE:-} ]]; then
    TRAIN_TEACHER_CACHE=${TRAIN_TEACHER_CACHE:-$TEACHER_CACHE/train}
    VAL_TEACHER_CACHE=${VAL_TEACHER_CACHE:-$TEACHER_CACHE/val}
fi

COMMON=(--vod-root "$VOD" --surface-checkpoint "$SURFACE"
        --batch-size "$BATCH_SIZE" --validation-batch-size "$BATCH_SIZE"
        --num-workers "$NUM_WORKERS" --seed "$SEED"
        --head-learning-rate "$HEAD_LEARNING_RATE"
        --feature-learning-rate "$FEATURE_LEARNING_RATE"
        --device cuda --no-tensorboard)
[[ -z ${FEATURE:-} ]] || COMMON+=(--feature-student-checkpoint "$FEATURE")
[[ -z ${TEACHER:-} ]] || COMMON+=(--teacher-checkpoint "$TEACHER")
[[ -z ${DATA_CACHE:-} ]] || COMMON+=(--data-cache-root "$DATA_CACHE")
[[ -z ${TRAIN_TEACHER_CACHE:-} ]] || COMMON+=(--training-cache-root "$TRAIN_TEACHER_CACHE")
[[ -z ${VAL_TEACHER_CACHE:-} ]] || COMMON+=(--validation-cache-root "$VAL_TEACHER_CACHE")

mkdir -p "$OUT"
for arm in control joint; do
    smoke="$OUT/smoke/$arm"
    if [[ -s "$smoke/last.ckpt" ]]; then
        echo "Smoke $arm already completed: $smoke/last.ckpt"
        continue
    fi
    echo "=== One-frame smoke: $arm ==="
    options=(--distillation-weight "$DISTILLATION_WEIGHT")
    if [[ $arm == control ]]; then
        options=(--freeze-feature-student --distillation-weight 0)
    fi
    "$PY" -u -m models.radar_lidar_crossmodal_v1.train_joint_surface_student \
        "${COMMON[@]}" "${options[@]}" --output-root "$smoke" \
        --epochs 1 --train-limit 1 --val-limit 1 \
        --quick-val-frames 1 --full-val-every 1
done

for arm in control joint; do
    run="$OUT/full/$arm"
    echo "=== Full ablation: $arm ==="
    options=(--distillation-weight "$DISTILLATION_WEIGHT")
    if [[ $arm == control ]]; then
        options=(--freeze-feature-student --distillation-weight 0)
    fi
    if [[ -s "$run/last.ckpt" ]]; then
        options+=(--resume "$run/last.ckpt")
    fi
    "$PY" -u -m models.radar_lidar_crossmodal_v1.train_joint_surface_student \
        "${COMMON[@]}" "${options[@]}" --output-root "$run" \
        --epochs "$EPOCHS" --quick-val-frames "$QUICK_VAL_FRAMES" \
        --full-val-every "$FULL_VAL_EVERY"
done

"$PY" -m scripts.compare_joint_surface_ablation \
    --control-root "$OUT/full/control" --joint-root "$OUT/full/joint" \
    --output "$OUT/ablation_comparison.json"
