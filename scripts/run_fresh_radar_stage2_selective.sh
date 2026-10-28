#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh official five-scan radar -> occupancy -> selective Stage-II completion.
# A new OUT gives new model checkpoints, gate caches, and fog injection data.
# Rerunning with the same OUT resumes only checkpoints made by this run.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
OUT=${OUT:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/radar_stage2_selective_20261028}
RAW=${RAW:-$OUT/fixed_frames}
WORKERS=${WORKERS:-8}
FOG_WORKERS=${FOG_WORKERS:-2}
LIDAR_ENCODER=${LIDAR_ENCODER:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/crossmodal_fov70_refined_noaug_30ep_20261015/full/lidar/best_encoder.pth}

RADAR_VARIANT=radar_5frames
NOAUG=$REPO/configs/crossmodal_no_geometric_augmentation.json
CONFIG=$REPO/configs/crossmodal_meta_occupancy.json
RADAR_RUN=$OUT/radar_pretrain
RADAR_ENCODER=$RADAR_RUN/best_encoder.pth
GEOMETRY=$OUT/meta_geometry_v3
STAGE1=$OUT/radar_autoencoder
STAGE1_CKPT=$STAGE1/best_student.ckpt
FOG_CACHE=$OUT/stage2_severe_fog_s5_v1
STAGE2_CACHE=$OUT/stage2_fixed_gate025_v1
LPCR=$OUT/stage2_sparse_lpcr_fullregion_v9
DIFF=$OUT/stage2_sparse_diffusion_selective_v10

fail() { printf '%s\n' "$*" >&2; exit 1; }
on_error() {
  local status=$?
  printf 'Fresh training stopped at line %s (exit %s). Outputs remain in %s\n' \
    "$1" "$status" "$OUT" >&2
  exit "$status"
}
trap 'on_error $LINENO' ERR

[[ -x "$PY" ]] || fail "Python environment missing: $PY"
[[ -s "$LIDAR_ENCODER" ]] || fail "Geometry reference LiDAR encoder missing: $LIDAR_ENCODER"
[[ -s "$NOAUG" && -s "$CONFIG" ]] || fail "Training configuration missing under $REPO/configs"
[[ -d "$VOD/radar_5frames/training/velodyne" ]] || fail "Official five-scan radar missing under $VOD"
[[ -s "$VOD/lidar/ImageSets/train.txt" && -s "$VOD/lidar/ImageSets/val.txt" ]] || fail "VoD train/val split missing"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && "$FOG_WORKERS" =~ ^[1-9][0-9]*$ ]] || fail "WORKERS and FOG_WORKERS must be positive integers"
command -v flock >/dev/null || fail "flock is required to prevent two runs writing the same output"
command -v timeout >/dev/null || fail "GNU timeout is required for the CUDA smoke check"

mkdir -p "$OUT" "$OUT/matplotlib"
exec 9>"$OUT/run.lock"
flock -n 9 || fail "Another process is using $OUT"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export MPLCONFIGDIR="$OUT/matplotlib" MPLBACKEND=Agg
cd "$REPO"

# The upstream simulator is an external, non-commercial research dependency.
# Keep it outside this source repository and retain its own LICENSE file.
lookup_name=integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_0.06.pickle
valid_fog_sim() {
  [[ -s "$1/fog_simulation.py" && -s "$1/integral_lookup_tables/original/$lookup_name" ]]
}
if [[ -z "${FOG_SIM:-}" ]]; then
  for candidate in \
    "$BASE/Sensor-Fusion/Weather_Injector/LiDAR_fog_sim" \
    "$BASE/Sensor-Fusion_Final_Model_Repo/Weather_Injector/LiDAR_fog_sim" \
    "$BASE/Weather_Injector/LiDAR_fog_sim" \
    "$OUT/dependencies/LiDAR_fog_sim"; do
    if valid_fog_sim "$candidate"; then FOG_SIM=$candidate; break; fi
  done
fi
if [[ -z "${FOG_SIM:-}" ]]; then
  command -v curl >/dev/null || fail "Set FOG_SIM to the original LiDAR_fog_sim directory; curl is unavailable"
  command -v tar >/dev/null || fail "Set FOG_SIM to the original LiDAR_fog_sim directory; tar is unavailable"
  FOG_SIM=$OUT/dependencies/LiDAR_fog_sim
  mkdir -p "$FOG_SIM"
  printf '\n=== Download original fog simulator and lookup table ===\n'
  curl -fL --retry 3 \
    https://github.com/MartinHahner/LiDAR_fog_sim/archive/1cb9e072976047a7d11f859797d939676d37d88f.tar.gz \
    -o "$OUT/dependencies/LiDAR_fog_sim.tar.gz"
  tar -xzf "$OUT/dependencies/LiDAR_fog_sim.tar.gz" \
    --strip-components=1 -C "$FOG_SIM"
fi
valid_fog_sim "$FOG_SIM" || fail "Missing fog_simulation.py or severity-5 lookup in $FOG_SIM; set FOG_SIM to a complete upstream checkout"

printf 'Run output: %s\nRadar: %s\nFog simulator: %s\n' "$OUT" "$RADAR_VARIANT" "$FOG_SIM"
printf 'Fog source SHA256: %s\n' "$(sha256sum "$FOG_SIM/fog_simulation.py" | cut -d' ' -f1)"
printf 'Fog lookup SHA256: %s\n' "$(sha256sum "$FOG_SIM/integral_lookup_tables/original/$lookup_name" | cut -d' ' -f1)"

# Catch a broken MinkowskiEngine/CUDA import before hours of cache creation.
timeout 180s "$PY" -u -m scripts.smoke_stage2_sparse --device cuda

printf '\n=== Fixed aligned VoD frames (train and validation) ===\n'
for split in train val; do
  "$PY" -u -m scripts.cache_crossmodal_fixed_frames \
    --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
    --cache-root "$RAW" --split "$split" --num-workers "$WORKERS"
done

printf '\n=== Fresh severity-5 fog cache, with original-return filtering ===\n'
"$PY" -u -m scripts.cache_stage2_severe_fog \
  --vod-root "$VOD" --fog-simulator-root "$FOG_SIM" \
  --cache-root "$FOG_CACHE" --radar-variant "$RADAR_VARIANT" \
  --split both --num-workers "$FOG_WORKERS" --max-cache-gib 5
"$PY" -u -m scripts.smoke_stage2_severe_faults \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --fault-cache-root "$FOG_CACHE" --split val

printf '\n=== Radar encoder pretraining from scratch (30 epochs) ===\n'
radar_resume=()
if [[ -s "$RADAR_RUN/last.ckpt" ]]; then
  radar_resume=(--resume "$RADAR_RUN/last.ckpt")
elif [[ -e "$RADAR_RUN/metrics.jsonl" || -e "$RADAR_ENCODER" ]]; then
  fail "Incomplete radar pretraining in $RADAR_RUN; inspect before resuming"
fi
"$PY" -u -m models.radar_lidar_crossmodal_v1.pretrain_encoder \
  --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
  --modality radar --data-cache-root "$RAW" --output-root "$RADAR_RUN" \
  --online-augmentation-config "$NOAUG" \
  --epochs 30 --batch-size 16 --full-val-every 5 \
  --num-workers "$WORKERS" --device cuda --no-tensorboard \
  "${radar_resume[@]}"
[[ -s "$RADAR_ENCODER" ]] || fail "Radar encoder pretraining produced no best checkpoint"

printf '\n=== Lossless Stage-I spatial geometry cache ===\n'
for split in train val; do
  "$PY" -u -m scripts.cache_meta_occupancy \
    --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
    --data-cache-root "$RAW" --cache-root "$GEOMETRY" \
    --radar-encoder "$RADAR_ENCODER" --lidar-encoder "$LIDAR_ENCODER" \
    --config "$CONFIG" --split "$split" --limit 2 \
    --num-workers 2 --device cuda
done
"$PY" -u -m scripts.smoke_meta_cache \
  --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
  --data-cache-root "$RAW" --cache-root "$GEOMETRY" \
  --radar-encoder "$RADAR_ENCODER" --lidar-encoder "$LIDAR_ENCODER" \
  --config "$CONFIG" --check-gradients --device cuda
for split in train val; do
  "$PY" -u -m scripts.cache_meta_occupancy_fast \
    --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
    --data-cache-root "$RAW" --cache-root "$GEOMETRY" \
    --radar-encoder "$RADAR_ENCODER" --lidar-encoder "$LIDAR_ENCODER" \
    --config "$CONFIG" --split "$split" --verify-existing 2 \
    --num-workers "$WORKERS"
done

printf '\n=== Fresh radar-only Stage I occupancy training (25 epochs) ===\n'
stage1_resume=()
if [[ -s "$STAGE1/last.ckpt" ]]; then
  stage1_resume=(--resume "$STAGE1/last.ckpt")
elif [[ -e "$STAGE1/metrics.jsonl" || -e "$STAGE1_CKPT" ]]; then
  fail "Incomplete Stage I state in $STAGE1; inspect before resuming"
fi
"$PY" -u -m models.radar_lidar_crossmodal_v1.train_radar_autoencoder \
  --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
  --radar-encoder "$RADAR_ENCODER" --data-cache-root "$RAW" \
  --meta-cache-root "$GEOMETRY" --output-root "$STAGE1" \
  --config "$CONFIG" --epochs 25 --batch-size 8 \
  --num-workers "$WORKERS" --scan-metrics-every 5 \
  --full-val-every 5 --device cuda "${stage1_resume[@]}"
[[ -s "$STAGE1_CKPT" ]] || fail "Stage I produced no best_student.ckpt"

printf '\n=== Fresh per-frame Stage-I gate at confidence 0.25 ===\n'
"$PY" -u -m scripts.cache_stage2_sparse \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --stage1-checkpoint "$STAGE1_CKPT" --cache-root "$STAGE2_CACHE" \
  --radar-variant "$RADAR_VARIANT" --gate-threshold 0.25 \
  --split both --num-workers "$WORKERS" --device cuda

printf '\n=== Full-region clean-LiDAR LPCR from scratch (30 epochs) ===\n'
lpcr_resume=()
if [[ -s "$LPCR/last.ckpt" ]]; then
  lpcr_resume=(--resume "$LPCR/last.ckpt")
elif [[ -e "$LPCR/metrics.jsonl" || -e "$LPCR/best.ckpt" ]]; then
  fail "Incomplete LPCR state in $LPCR; inspect before resuming"
fi
"$PY" -u -m scripts.train_stage2_sparse \
  --phase lpcr --vod-root "$VOD" --data-cache-root "$RAW" \
  --output-root "$LPCR" --radar-variant "$RADAR_VARIANT" \
  --epochs 30 --lr 0.001 --batch-size 8 --num-workers "$WORKERS" \
  --paper-metrics-every 5 --paper-metrics-frames 4 \
  --device cuda "${lpcr_resume[@]}"
[[ -s "$LPCR/best.ckpt" ]] || fail "LPCR produced no best.ckpt"

printf '\n=== Selective-addition sparse residual diffusion (50 epochs) ===\n'
diff_resume=()
if [[ -s "$DIFF/last.ckpt" ]]; then
  diff_resume=(--resume "$DIFF/last.ckpt")
elif [[ -e "$DIFF/metrics.jsonl" || -e "$DIFF/best.ckpt" ]]; then
  fail "Incomplete diffusion state in $DIFF; inspect before resuming"
fi
"$PY" -u -m scripts.train_stage2_sparse \
  --phase diffusion --vod-root "$VOD" \
  --data-cache-root "$RAW" --stage2-cache-root "$STAGE2_CACHE" \
  --fault-cache-root "$FOG_CACHE" --stage1-checkpoint "$STAGE1_CKPT" \
  --lpcr-checkpoint "$LPCR/best.ckpt" --output-root "$DIFF" \
  --radar-variant "$RADAR_VARIANT" --gate-threshold 0.25 \
  --epochs 50 --lr 0.0002 --batch-size 16 --num-workers "$WORKERS" \
  --residual-weight 1 --reconstruction-weight 1 \
  --addition-weight 1 --healthy-weight 1 \
  --ray-hit-weight 0 --ray-extra-weight 0 \
  --device cuda "${diff_resume[@]}"
[[ -s "$DIFF/best.ckpt" ]] || fail "Diffusion produced no best.ckpt"

printf '\nComplete. Stage I: %s\nLPCR: %s\nDiffusion: %s\n' \
  "$STAGE1_CKPT" "$LPCR/best.ckpt" "$DIFF/best.ckpt"
