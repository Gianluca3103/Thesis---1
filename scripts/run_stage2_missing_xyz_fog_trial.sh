#!/usr/bin/env bash
# Rebuild mild fog without intensity-based deletion, then train a separate BCE trial.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
FOG=${FOG:-$BASE/Sensor-Fusion/Weather_Injector/LiDAR_fog_sim}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
MILD=${MILD:-$DATA/stage2_mild_fog2_xyz_presence_v2_20261108}
FAULT_REGIONS=${FAULT_REGIONS:-$DATA/stage2_mild_fog2_xyz_presence_v2_regions_20261108}
OUT=${OUT:-$DATA/stage2_conditional_generated_bce_missing_xyz_v2_20261108}
WORKERS=${WORKERS:-2}

for required in "$PY" "$FOG/fog_simulation.py" \
  "$FOG/integral_lookup_tables/original/integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_0.01.pickle" \
  "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$RAW/test/manifest.json"; do
  [[ -s "$required" ]] || { printf 'Missing prerequisite: %s\n' "$required" >&2; exit 1; }
done

cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
mkdir -p "$DATA/stage2_clean_tmp"
export TMPDIR="$DATA/stage2_clean_tmp"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for split in both test; do
  printf 'Caching %s mild faults with original-XYZ presence policy.\n' "$split"
  "$PY" -u -m scripts.cache_stage2_mild_fog2 \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --fog-simulator-root "$FOG" --output-root "$MILD" \
    --split "$split" --num-workers "$WORKERS"
done

printf 'Training a separate Stage II run on the corrected cache.\n'
REPO="$REPO" BASE="$BASE" DATA="$DATA" PY="$PY" VOD="$VOD" \
  RAW="$RAW" SUPPORT="$SUPPORT" MILD="$MILD" \
  FAULT_REGIONS="$FAULT_REGIONS" OUT="$OUT" WORKERS="$WORKERS" \
  GENERATED_BCE=1 BCE_LOSS_WEIGHT=0.1 BATCH_SIZE=16 EPOCHS=75 \
  bash scripts/run_stage2_conditional_from_scratch.sh
