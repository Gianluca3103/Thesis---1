#!/usr/bin/env bash
# Build severity-2 mixed faults, train a fresh fault-aware Stage I, evaluate held-out.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-hierarchical-stage1-20261101}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
FOG_SIM=${FOG_SIM:-$BASE/Sensor-Fusion/Weather_Injector/LiDAR_fog_sim}
CONFIG=${CONFIG:-$REPO/configs/hierarchical_stage1_gate_expanded_v1.json}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
OUT=${OUT:-$DATA/hierarchical_stage1_mild_fog2_20261101}
OLD=${OLD:-$DATA/hierarchical_stage1_expanded_full_20261101/best.ckpt}
WORKERS=${WORKERS:-4}
VARIANT=radar_20frames_verified_doppler_radial

fail() { printf '%s\n' "$*" >&2; exit 1; }
for required in "$PY" "$CONFIG" "$FOG_SIM/fog_simulation.py" \
  "$FOG_SIM/integral_lookup_tables/original/integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_0.01.pickle" \
  "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$REPO/scripts/cache_stage2_mild_fog2.py" \
  "$REPO/scripts/train_hierarchical_stage1.py"; do
  [[ -s "$required" ]] || fail "Missing required file: $required"
done
[[ -d "$VOD" ]] || fail "Missing VoD directory: $VOD"
[[ "$MILD" != "$OUT" && "$MILD" != "$RAW" && "$OUT" != "$RAW" ]] || fail "Use separate cache and training output directories"
[[ "$WORKERS" =~ ^[0-9]+$ ]] || fail "WORKERS must be a nonnegative integer"

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is required for full Stage-I training"
print(f"Torch {torch.__version__}; CUDA {torch.cuda.get_device_name(0)}", flush=True)
PY

# Build a few validation entries first, then finish both splits in the same cache.
"$PY" -u -m scripts.cache_stage2_mild_fog2 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --fog-simulator-root "$FOG_SIM" --output-root "$MILD" \
  --radar-variant "$VARIANT" --split val --limit 16 \
  --num-workers 2 --max-cache-gib 10
"$PY" -u -m scripts.cache_stage2_mild_fog2 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --fog-simulator-root "$FOG_SIM" --output-root "$MILD" \
  --radar-variant "$VARIANT" --split both \
  --num-workers 2 --max-cache-gib 10

"$PY" - "$MILD" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np
root = Path(sys.argv[1])
for split in ("train", "val"):
    folder = root / split
    manifest = json.loads((folder / "manifest.json").read_text())
    missing = [frame for frame in manifest["frame_ids"] if not (folder / f"{frame}.npz").is_file()]
    if missing:
        raise SystemExit(f"Incomplete {split} cache: {len(missing)} missing, first {missing[0]}")
    ratios = {name: [] for name in manifest["fault_types"]}
    for frame in manifest["frame_ids"]:
        with np.load(folder / f"{frame}.npz", allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
        if (metadata["version"] != manifest["version"]
                or metadata["source_manifest_sha256"] != manifest["source_manifest_sha256"]
                or metadata["frame_id"] != frame
                or metadata["fault_type"] != manifest["frame_faults"][frame]):
            raise SystemExit(f"Incorrect {split} cache entry: {frame}")
        ratios[metadata["fault_type"]].append(metadata["retained_fraction_actual"])
    print(f"{split}: {len(manifest['frame_ids'])} complete frames")
    for name, values in ratios.items():
        print(f"  {name}: {len(values)} frames, median retained {np.median(values):.1%}, "
              f"10th percentile {np.quantile(values, .1):.1%}")
PY

"$PY" -u -m scripts.visualize_stage2_mild_faults \
  --mild-root "$MILD" --output-root "$MILD/views_val" \
  --split val --per-fault 2

if [[ ! -s "$OUT/smoke/best.ckpt" ]]; then
  "$PY" -u -m scripts.train_hierarchical_stage1 \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --faulty-root "$MILD" --fault-modes mild_dataset \
    --radar-variant "$VARIANT" --output-root "$OUT/smoke" --config "$CONFIG" \
    --train-limit 1 --val-limit 2 --epochs 1 --num-workers 0 \
    --budgets 500 --device cuda 2>&1 | tee "$OUT/smoke.log"
fi
[[ -s "$OUT/smoke/best.ckpt" ]] || fail "Stage-I smoke checkpoint was not created"

resume=()
if [[ -s "$OUT/last.ckpt" ]]; then
  resume=(--resume "$OUT/last.ckpt")
elif [[ -e "$OUT/metrics.jsonl" ]]; then
  fail "Training metrics exist without a resumable checkpoint in $OUT"
fi
"$PY" -u -m scripts.train_hierarchical_stage1 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --faulty-root "$MILD" --fault-modes mild_dataset \
  --radar-variant "$VARIANT" --output-root "$OUT" --config "$CONFIG" \
  --epochs 25 --val-limit 128 --val-selection blocks \
  --num-workers "$WORKERS" --seed 3103 \
  --budgets 250 500 1000 2000 5000 \
  --device cuda "${resume[@]}" 2>&1 | tee -a "$OUT/train.log"
[[ -s "$OUT/best.ckpt" ]] || fail "Full Stage-I checkpoint was not created"

common=(--vod-root "$VOD" --faulty-root "$MILD"
  --evaluate-modes mild_dataset --limit 512 --seed 3103
  --budgets 250 500 1000 2000 5000 --device cuda)
"$PY" -u -m scripts.compare_hierarchical_stage1 "${common[@]}" \
  --checkpoint "$OUT/best.ckpt" --output "$OUT/heldout_mild_fog2.json" \
  2>&1 | tee "$OUT/heldout_mild_fog2.log"
OLD_VARIANT=""
if [[ -s "$OLD" ]]; then
  OLD_VARIANT=$("$PY" -c 'import sys,torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["radar_variant"])' "$OLD")
fi
if [[ -s "$OLD" && "$OLD_VARIANT" == "$VARIANT" ]]; then
  "$PY" -u -m scripts.compare_hierarchical_stage1 "${common[@]}" \
    --checkpoint "$OLD" --output "$OUT/old_stage1_on_mild_fog2.json" \
    2>&1 | tee "$OUT/old_stage1_on_mild_fog2.log"
  "$PY" - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
new = json.loads((root / "heldout_mild_fog2.json").read_text())
old = json.loads((root / "old_stage1_on_mild_fog2.json").read_text())
if new["sampled_frame_ids"] != old["sampled_frame_ids"]:
    print("Old/new evaluation frames differ; compare reports separately.")
else:
    print("Matched old/new held-out Stage-I results on mild-fog2:")
    for budget in ("250", "500", "1000", "2000", "5000"):
        a = old["heldout"]["hierarchical"]["mild_dataset"][budget]
        b = new["heldout"]["hierarchical"]["mild_dataset"][budget]
        print(f"  top {budget}: old P={a['precision']:.1%} R={a['recall']:.1%}; "
              f"new P={b['precision']:.1%} R={b['recall']:.1%}")
PY
fi
if [[ -s "$OLD" && "$OLD_VARIANT" != "$VARIANT" ]]; then
  printf 'Skipping old Stage-I comparison: radar variant is %s, expected %s\n' "$OLD_VARIANT" "$VARIANT" >&2
fi
printf 'New dataset: %s\nNew Stage-I checkpoint: %s\nViewer: %s\nHeld-out report: %s\n' \
  "$MILD" "$OUT/best.ckpt" "$MILD/views_val/index.html" "$OUT/heldout_mild_fog2.json"
