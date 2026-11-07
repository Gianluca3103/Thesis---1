#!/usr/bin/env bash
# Cache the >=12 radar-voxel/1.6 m gate and ray labels, then train Stage II.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
REGIONS=${REGIONS:-$DATA/stage2_mild_fog2_injector_regions_20261106}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
RAY_LABELS=${RAY_LABELS:-$DATA/stage2_ray_labels12vox_1p6m_mild_fog2_20261107}
OUT=${OUT:-$DATA/stage2_geometry12vox_1p6m_epsilon_ray_offset_mild_batch8_20261107}
LPCR=${LPCR:-$DATA/radar_stage2_ray0_20261028/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
if [[ ! -s "$LPCR" ]]; then
  LPCR=$DATA/stage2_fullregion_lpcr_for_geometry_20261106/best.ckpt
fi

for required in "$PY" "$LPCR" \
  "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$MILD/train/manifest.json" "$MILD/val/manifest.json" \
  "$REGIONS/train/manifest.json" "$REGIONS/val/manifest.json" \
  "$REGIONS/train/regions.json" "$REGIONS/val/regions.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done

mkdir -p "$OUT" "$SUPPORT" "$RAY_LABELS" "$DATA/stage2_geometry_tmp"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export TMPDIR="$DATA/stage2_geometry_tmp"
export STAGE2_RAY_WORKERS=${RAY_WORKERS:-8}
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

printf '1/4: precompute 12-radar-voxel/1.6 m support masks.\n'
"$PY" -u -m scripts.cache_radar_pruner_stage1 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --output-root "$SUPPORT" --split both --radius-m 1.6 \
  --min-distinct-radar-voxels 12 --num-workers 2 \
  2>&1 | tee "$OUT/cache_support.log"

printf '2/4: run a one-epoch GPU smoke test on 32 train and 8 validation frames.\n'
if [[ ! -s "$OUT/smoke/last.ckpt" ]]; then
  REPO="$REPO" BASE="$BASE" DATA="$DATA" PY="$PY" VOD="$VOD" RAW="$RAW" \
  MILD="$MILD" REGIONS="$REGIONS" SUPPORT="$SUPPORT" RAY_LABELS= \
  LPCR="$LPCR" OUT="$OUT/smoke" MIN_RADAR_VOXELS=12 \
  WORKERS=2 RAY_WORKERS=${RAY_WORKERS:-8} BATCH_SIZE=8 \
  TRAIN_LIMIT=32 VAL_LIMIT=8 EPOCHS=1 FULL_VAL=0 EVAL=0 \
  bash scripts/run_stage2_geometry_epsilon_ray_offset_mild.sh
fi

printf '3/4: precompute fixed ray labels for this exact candidate set.\n'
"$PY" -u -m scripts.cache_stage2_ray_labels \
  --vod-root "$VOD" --data-cache-root "$RAW" --faulty-root "$MILD" \
  --support-cache-root "$SUPPORT" --fault-region-root "$REGIONS" \
  --output-root "$RAY_LABELS" --geometry-min-radar-voxels 12 \
  --split both --device cuda \
  2>&1 | tee "$OUT/cache_rays.log"

printf '4/4: train epsilon/ray/offset Stage II and evaluate its best checkpoint.\n'
REPO="$REPO" BASE="$BASE" DATA="$DATA" PY="$PY" VOD="$VOD" RAW="$RAW" \
MILD="$MILD" REGIONS="$REGIONS" SUPPORT="$SUPPORT" RAY_LABELS="$RAY_LABELS" \
LPCR="$LPCR" OUT="$OUT" MIN_RADAR_VOXELS=12 \
WORKERS=${WORKERS:-2} RAY_WORKERS=${RAY_WORKERS:-8} BATCH_SIZE=${BATCH_SIZE:-8} \
EPOCHS=${EPOCHS:-25} FULL_VAL=1 EVAL=1 \
bash scripts/run_stage2_geometry_epsilon_ray_offset_mild.sh

printf 'Checkpoint: %s\n' "$OUT/best.ckpt"
printf 'Held-out sweep: %s\n' "$OUT/validation_512_calibrated.json"
