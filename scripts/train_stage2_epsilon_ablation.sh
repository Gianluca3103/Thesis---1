#!/usr/bin/env bash
set -Eeuo pipefail

# Epsilon-prediction Stage-II ablation. Reuse the same Stage-I gate, faults,
# and full-region LPCR as the residual run; save a distinct diffusion model.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
SOURCE_RUN=${SOURCE_RUN:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/radar_stage2_ray0_20261028}
OUT=${OUT:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/radar_stage2_epsilon_ray025_20261029}
WORKERS=${WORKERS:-8}

STAGE1=$SOURCE_RUN/radar_autoencoder/best_student.ckpt
LPCR=$SOURCE_RUN/stage2_sparse_lpcr_fullregion_v9/best.ckpt
RAW=$SOURCE_RUN/fixed_frames
GATE=$SOURCE_RUN/stage2_fixed_gate025_v1
FOG=$SOURCE_RUN/stage2_severe_fog_s5_v1
DIFF=$OUT/stage2_sparse_diffusion_epsilon_ray025_v1

for path in "$PY" "$STAGE1" "$LPCR" \
  "$GATE/train/manifest.json" "$GATE/val/manifest.json" \
  "$FOG/train/manifest.json" "$FOG/val/manifest.json"; do
  [[ -s "$path" ]] || { printf 'Missing required file: %s\n' "$path" >&2; exit 1; }
done
[[ -d "$RAW/train" && -d "$RAW/val" ]] || {
  printf 'Fixed-frame cache is missing: %s\n' "$RAW" >&2; exit 1;
}
command -v flock >/dev/null || { echo 'flock is required' >&2; exit 1; }
mkdir -p "$OUT" "$OUT/matplotlib"
exec 9>"$OUT/run.lock"
flock -n 9 || { echo "Another process is writing to $OUT" >&2; exit 1; }

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export MPLCONFIGDIR="$OUT/matplotlib" MPLBACKEND=Agg
cd "$REPO"

"$PY" -u -m scripts.smoke_stage2_sparse --device cuda --check-epsilon

resume=()
if [[ -s "$DIFF/last.ckpt" ]]; then
  resume=(--resume "$DIFF/last.ckpt")
elif [[ -e "$DIFF/metrics.jsonl" || -e "$DIFF/best.ckpt" ]]; then
  echo "Incomplete epsilon run in $DIFF; inspect before resuming" >&2
  exit 1
fi

"$PY" -u -m scripts.train_stage2_sparse \
  --phase diffusion --diffusion-target epsilon \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --stage2-cache-root "$GATE" --fault-cache-root "$FOG" \
  --stage1-checkpoint "$STAGE1" --lpcr-checkpoint "$LPCR" \
  --output-root "$DIFF" --radar-variant radar_5frames \
  --gate-threshold 0.25 --epochs 50 --lr 0.0002 \
  --batch-size 16 --num-workers "$WORKERS" \
  --residual-weight 0 --reconstruction-weight 1 \
  --addition-weight 1 --healthy-weight 1 \
  --ray-hit-weight 0.25 --ray-extra-weight 0.25 \
  --device cuda "${resume[@]}"

printf 'Epsilon checkpoint: %s\n' "$DIFF/best.ckpt"
