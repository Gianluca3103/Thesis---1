#!/usr/bin/env bash
# Evaluate the completed batch-eight mild-fault run and make balanced previews.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
SUPPORT=${SUPPORT:-$DATA/radar_pruner3vox_1p6m_mild_fog2_20261103/support_masks}
REGIONS=${REGIONS:-$DATA/stage2_mild_fog2_injector_regions_20261106}
OUT=${OUT:-$DATA/stage2_geometry1p6m_injector_region_epsilon_ray_offset_mild_batch8_20261107}

for required in "$PY" "$OUT/last.ckpt" "$OUT/best.ckpt" \
  "$RAW/val/manifest.json" "$MILD/val/manifest.json" \
  "$SUPPORT/val/manifest.json" "$REGIONS/val/manifest.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PY" - "$OUT" <<'PY'
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
last = torch.load(root / "last.ckpt", map_location="cpu", weights_only=False)
best = torch.load(root / "best.ckpt", map_location="cpu", weights_only=False)
if last.get("epoch") != 25:
    raise SystemExit(f"Training has not finished epoch 25: last.ckpt is epoch {last.get('epoch')}")
print(f"Training checkpoint: epoch {last['epoch']}; best loss checkpoint: epoch {best['epoch']}", flush=True)
PY

"$PY" -u -m scripts.evaluate_stage2_geometry_baseline \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --faulty-root "$MILD" --support-cache-root "$SUPPORT" \
  --fault-region-root "$REGIONS" --checkpoint "$OUT/best.ckpt" \
  --output "$OUT/validation_512_calibrated_post25.json" \
  --limit 512 --sampling-steps 64 --device cuda \
  2>&1 | tee "$OUT/evaluate_post25.log"

"$PY" -u -m scripts.predict_stage2_sparse \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --faulty-root "$MILD" --support-cache-root "$SUPPORT" \
  --fault-region-root "$REGIONS" --stage2-checkpoint "$OUT/best.ckpt" \
  --radar-variant radar_20frames_verified_doppler_radial \
  --split val --skip-first 64 --per-fault 2 \
  --sampling-steps 64 --occupancy-threshold 0.5 \
  --output-root "$OUT/val_samples_post25" --html --device cuda \
  2>&1 | tee "$OUT/visualize_post25.log"

# The current run used the former 64-frame validation setting. Score the
# best checkpoint on all eligible validation cells without training again.
"$PY" -u -m scripts.evaluate_stage2_eligible_occupancy \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --faulty-root "$MILD" --support-cache-root "$SUPPORT" \
  --fault-region-root "$REGIONS" --checkpoint "$OUT/best.ckpt" \
  --output "$OUT/eligible_occupancy_best_post25.json" \
  --sampling-steps 64 --occupancy-threshold 0.5 --device cuda \
  2>&1 | tee "$OUT/eligible_occupancy_post25.log"

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sweep = json.loads((root / "validation_512_calibrated_post25.json").read_text())
heldout = sweep["splits"]["heldout"]
print(f"Sweep: best checkpoint epoch {sweep['checkpoint_epoch']}; "
      f"{heldout['frames']} held-out frames", flush=True)
for name, row in heldout["calibration_selected"].items():
    if row is not None:
        print(f"  {name}: threshold={row['threshold']:.4f}, "
              f"precision={row['precision']:.2%}, "
              f"eligible recall={row['eligible_recall']:.2%}, "
              f"F1={row['f1']:.2%}", flush=True)
eligible = json.loads((root / "eligible_occupancy_best_post25.json").read_text())
print(f"Best checkpoint epoch {eligible['checkpoint_epoch']}, "
      f"{eligible['frames']} eligible frames: "
      f"exact IoU={eligible['eligible_iou']:.2%}, "
      f"IoU@0.2m={eligible['eligible_iou_0p2m']:.2%}", flush=True)
print(f"View samples: {root / 'val_samples_post25' / 'index.html'}", flush=True)
PY
