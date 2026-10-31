# Thesis-1: radar–LiDAR cross-modal learning

Standalone encoder → cross-modal teacher → radar-only student pipeline,
based on the native sparse Stage-I backbone used in the earlier interpolation
views. Datasets, trained weights,
older experiments, and generated reports are excluded.

## What is here

| Path | Purpose |
| --- | --- |
| `models/radar_lidar_crossmodal_v1/` | Independent sparse-encoder pretraining, frozen-encoder teacher, radar-only student, evaluation, raw export |
| `models/radar_lidar_stage1/sparse.py` | Native Stage-I radar and LiDAR point encoders and four-scale sparse backbone |
| `models/radar_lidar_crossmodal_v1/vendor/` | Retained Downloads encoder source, inactive in this pipeline |
| `Fault_Localization_Model/vod_dataset/` | VoD decoding, calibration, verified 20-scan radar accumulation, Doppler radial compensation |
| `models/radar_lidar_stage1/` | Shared voxel, sparse, correspondence, paired-data, and augmentation utilities |
| `models/radar_lidar_stage2/` | Minimal candidate-domain contract and measured/free-space target construction for the student's confidence loss |
| `scripts/` | Verified stack generation, Doppler radial variant generation, GPU decoder smoke check |
| `configs/` | Teacher, student, and online augmentation settings |

The screenshot's Stage-I checkpoint used a native sparse backbone with 16, 32,
64 and 128 feature channels and axial support growth at level 4. A trained
128→64 adapter preserves the existing 64-channel cross-attention and student
interface. Radar uses RCS, radial and compensated Doppler velocity, and scan
age; LiDAR uses reflectivity. Both encoders are retrained independently. The
screenshot also used a surface proposal head and interpolation; the backbone
alone does not reproduce those five clouds. The student still outputs raw
fine-voxel confidence.
The student's fine-grid output is **raw confidence logits and probabilities**, with no
interpolation or spherical influence expansion. A separate sparse Stage II
implementation now consumes that map; see
[Stage II](models/radar_lidar_stage2/README.md). Its weights require a new run.

See [the pipeline guide](models/radar_lidar_crossmodal_v1/README.md) for
losses, metrics, checkpoint formats, and individual training commands.

## Independent R2LDM LiDAR reconstruction baseline

`models/r2ldm_lidar_lpcr/` contains a separate implementation of the LiDAR
encoder and LPCR point decoder from R2LDM's first stage. It uses R2LDM's
`0.1 × 0.1 × 0.15 m` grid over `x=[0,32]`, `y=[-16,16]`, `z=[-2,4]`,
and trains only on VoD LiDAR. It does not read the cross-modal caches or change
the existing radar occupancy model. See its
[README](models/r2ldm_lidar_lpcr/README.md) for training and reconstruction.

For a full run, `scripts/prepare_and_train_crossmodal.sh` checks
the complete train/val verified 20-scan Doppler radar cache, fills missing
derived frames when possible, then trains the radar and LiDAR encoders in
parallel before the teacher and student phases. It writes separate encoder
checkpoints and logs under `OUT`. Online geometric augmentation changes the
training geometry each epoch. Once the teacher is trained, fixed validation
geometry and teacher correspondence can be prebuilt with
`scripts/train_crossmodal_student_cached.sh`; training supervision remains online.

## Prerequisites

Use a CUDA-capable PyTorch environment with NumPy, SciPy, tqdm, matplotlib,
and MinkowskiEngine. TensorBoard is optional with
`--no-tensorboard`. The existing professor-machine Python environment has
previously passed one-frame integration checks with the Downloads PFE;
the native Stage-I encoder architecture requires fresh GPU smoke tests.
No dataset or weights are stored in this repository.

The VoD root supplied as `--vod-root` must contain official `lidar`, `radar`,
`radar_5frames`, calibrations, labels, and `lidar/ImageSets` splits. If the
derived `radar_20frames_verified_doppler_radial` tree already exists, use it.
Otherwise, create it from the official scans and five-scan alignment:

```bash
VOD=/path/to/view_of_delft_PUBLIC
PY=/path/to/cuda/python
cd /path/to/Thesis---1
export PYTHONPATH="$PWD"

for split in train val; do
  "$PY" -m scripts.generate_vod_accumulated_radar \
    --vod-root "$VOD" --stack-sizes 20 --split "$split" \
    --alignment-source official-5 --output-suffix verified \
    --isolate-split-history --no-basic-validity-filter --num-workers 4
done

for split in train val; do
  "$PY" -m scripts.generate_vod_doppler_radar \
    --vod-root "$VOD" --split "$split" \
    --output-variant radar_20frames_verified_doppler_radial \
    --frame-period-s 0.10022 --doppler-sign 1 --num-workers 4
done

"$PY" -m scripts.smoke_crossmodal_raw_decoder
```

The 0.10022 s frame period is an explicit approximation because the stack's
seventh column records ordinal scan age. Both generators write derived files
inside the VoD root and refuse to silently replace existing outputs. Set
`--overwrite` only when intentionally rebuilding a variant.

## Training status

### Fine radar-to-LiDAR association trial

`scripts/run_fine_association_stage2.sh` initializes a separate Stage-I
association model from the trained cross-modal radar student. Every active
radar voxel on the 0.2 × 0.2 × 0.25 m grid has up to 32 learned pointer slots
within a **Euclidean 3 m radius**. During training, each slot has a nearest
clean-LiDAR voxel target within that radius when one exists. The pointer
confidence loss uses observed occupancy and verified free space. At inference
the model receives radar only; clean LiDAR is never used to choose candidate
coordinates. Duplicate slots from one radar voxel count once. Distinct radar
voxels combine their probabilities with noisy OR, so repeated support raises
the probability of a unique target voxel. If an earlier trained eight-slot
checkpoint is available, the runner copies its radar representation and first
eight slots before training the additional 24; otherwise it initializes from
the cross-modal radar student. The eight-slot checkpoints remain loadable for
comparison. Stage II reads the full merged map
and applies its gate threshold before allocating active reconstruction sites.

The trial writes new Stage-I weights, a checkpoint-specific Stage-II gate
cache, and new diffusion weights. It reuses aligned raw frames, the full-region
LiDAR LPCR checkpoint, and the existing severe-fault cache. The original
cross-modal student and its checkpoints remain available for comparison.

`scripts.compare_fine_association_edge` is an inference-only Stage-I
postprocessing comparison for the 32-slot checkpoint. It interpolates between
high-confidence candidate voxels only when nearby radar features and scores
agree, then removes any resulting voxel more than 3 m from an active fine
radar voxel. It exports raw and edge-aware views plus exact and 0.2 m
  clean-scan precision, recall, IoU and F1 for 1 m and 3 m neighborhoods around
  active radar voxel centers, plus the full forward-FOV grid as context. The
  `threshold_sweep.csv` file reports exact and one-to-one 0.2 m tolerant scores
  across confidence thresholds for raw candidates, edge-aware candidates at
  the same threshold, and edge-aware candidates matched to the raw cell count
  within each scope. Local recall counts every clean voxel in that radar
  neighborhood, including voxels the model did not propose. The HTML preview
  uses `--threshold` (0.25 by default). The sweep displays a frame progress bar
  and then prints only F1-optimal thresholds with precision, recall, IoU, and
  F1 for exact and 0.2 m matches. Use `--report-only path/to/summary.json` to
  print the best results from a completed run without processing frames again.
  GPU inference accepts `--batch-size` (default 4) and `--num-workers`
  (default 2); postprocessing and metric matching remain frame-local to avoid
  matching points from different frames. The progress bar counts batches.
  Visualized frames additionally show fixed-threshold 0.2 m audit panels for
  raw and edge predictions within `--audit-scope` (1 m radar support by
  default): green is a one-to-one clean match, red is an unmatched prediction,
  and blue is a missed clean voxel. Set `--threshold` to the sweep-selected
  value when comparing a visual sample with the reported score.
  It does not alter the Stage-II gate or diffusion checkpoint.

Phase 0 uses a masked child-distribution objective to train two independent
native Stage-I encoders and their 128→64 adapters. The loss mask now matches
the masked input's stride-8 cells. The teacher and student use only the saved
encoder weights. No trained weights are included; old Downloads-PFE teacher
and student checkpoints cannot be resumed with this encoder architecture.

## Provenance

Copied modules keep their original import paths so the pipeline can be moved
without changing checkpoint keys. The standalone `candidate_domain.py` keeps
only the immutable domain and center-coordinate API used by the current
student. Legacy interpolation and older Stage I model imports were removed
from that support module. The original repository is unchanged.
