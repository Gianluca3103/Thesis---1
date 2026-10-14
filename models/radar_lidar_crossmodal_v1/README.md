# Exact-PFE cross-modal teacher and radar-only student

This version is separate from `radar_lidar_stage1`, `radar_lidar_stage2`, and
their existing checkpoints. The exact `Downloads/encoder.py` is copied unchanged
into `vendor/`. Its matching `Downloads/config.py` changes only the requested
voxel size; the encoder architecture and 64-channel bottleneck are unchanged. The original
`PFE_Engine.forward` converts its output to a dense volume. The wrapper in
`encoder.py` invokes the original voxelizer, input MLP, and sparse backbone
and retains the stride-8 sparse result. It does not alter the vendored source.

The encoder uses 64 bottleneck channels and now uses a 0.2 × 0.2 × 0.25 m base
voxel, so the physical stride-8 cell is 1.6 × 1.6 × 2 m. The unchanged
point-cloud range yields a fine grid of 512 × 512 × 16 cells in XYZ order and
a stride-8 grid of 64 × 64 × 2 cells. Existing checkpoints and cached geometry
from the earlier 0.1 × 0.1 × 0.5 m grid must not be resumed for this grid.
Four attention heads therefore have 16 channels each. The vendor sources have
SHA256 values `e9090b89276322873b2027a329bd706a7661ea65e6f093ac1419c3ad1a016b89`
and `5a83fe1b302653796bdee062daeb72ff14bbd2d8795c4e1f72f8c0ea33b07023` for
the voxel-size-adjusted config.

## Phases and parameters

| Phase | Frozen | Trained | Output |
| --- | --- | --- | --- |
| 0 radar | none | independent original radar PFE and a temporary 512-bin head | `radar/best_encoder.pth` |
| 0 LiDAR | none | independent original LiDAR PFE and a temporary 512-bin head | `lidar/best_encoder.pth` |
| 1 teacher | both original PFEs | separate residual adapters, bottleneck Q/K/V/O, scalar gate, alignment projections | radar-coordinate LiDAR-aware 64D features |
| 2 student | radar PFE and complete teacher | shared residual transform, feature MLP, three-level generative sparse confidence decoder | radar-only 64D features and raw fine-voxel confidence |

No pretrained encoder weights were available for the exact Downloads encoder.
Phase 0 therefore trains two independent copies. Its **chosen new pretraining
objective** masks input features in complete stride-8 cells and predicts the
distribution of observed points over their 512 fine child cells. It does not
assert that unmeasured cells are empty and does not itself learn radar–LiDAR
correspondence. Sparse input coordinates remain present during masking, so this
is a feature-masking objective rather than a fully hidden geometry task. The
current pretrainer also applies its mask helper to already downsampled output
coordinates, dividing those coordinates by eight a second time. Consequently,
the sites selected for loss do not consistently match the feature-masked sites.
This requires a separate correction before Phase 0 learning curves can validate
the intended objective; the voxel-size-only change here does not alter it. The
512-bin head is discarded. `best_encoder.pth`
contains the original `PFE_Engine` state dictionary directly and is loaded
strictly by Phase 1. The original encoder source and architecture remain unchanged.

The PFE voxelizer uses XYZ mean and log point density. Although the input is the
aligned 20-frame radar stack, this exact encoder does **not** use Doppler, RCS,
or per-point age as network features. Current-frame radar returns are used for
reliable object-label assignment; the full stack supplies encoder geometry.

### Teacher

At stride 8, each radar voxel queries at most 24 LiDAR voxels within 6 m in
the same frame. Separate zero-initialized residual adapters feed four-head
local Q/K/V attention. A scalar gate begins near sigmoid(-2). The final tensor
is `LayerNorm(original_radar_feature + gate × projected_LiDAR_context)` at
the original radar coordinates. LiDAR features and annotations are never
coordinates in the exported teacher tensor.

The four configurable teacher losses are:

1. Object attention: `-mean(log(sum attention on same-instance candidates))`
   over labeled radar queries with a positive candidate.
2. Geometry: cross entropy from a normalized Gaussian over compatible local
   candidates, over all geometrically eligible scene queries. Its distance
   target is a pseudo-correspondence, not proof of a physical reflector match.
3. Contrastive: radar embeddings versus frozen-scene LiDAR object prototypes,
   using verified different instances as negatives.
4. Fusion: Smooth L1 to a stop-gradient, geometrically weighted combination
   of frozen LiDAR features, normalized to the teacher tensor's scale.

Default weights are 1, 0.5, 0.1, and 0.5 respectively; set a weight to zero
in the teacher JSON for a future ablation. `adapters_enabled=false` selects
direct Q/K/V from original frozen features. No ablation has been run.

Object labels use oriented VoD camera boxes projected from calibrated LiDAR
coordinates. Underlying points vote for a bottleneck voxel identity; a label
requires at least 80% dominance. Mixed or overlapping boxes are excluded.
Only reliably labeled radar queries with a same-instance LiDAR candidate enter
object Recall@1/3/5. The denominator is the count of these eligible queries;
candidate coverage is eligible divided by all reliably labeled queries.
Unannotated objects are never verified negatives. Positive/negative cosine
and their gap use the object-prototype contrastive pairs. Fusion MAE uses the
same normalized frozen reference as the fusion loss. The 2 m bottleneck height
is coarse and limits object-level separation.

### Student

The student's `forward(radar_points, radar_valid)` reads radar only. A frozen
original radar PFE feeds a trainable shared residual transform and two heads.
One head distills the teacher's 64D radar-coordinate tensor with Smooth L1.
The other expands each active stride-8 site through three MinkowskiEngine
generative transposed convolutions to fine stride 1 and predicts one raw logit
for **every generated site**. Candidate coordinates depend only on radar and
the fixed grid. No top-K, threshold pruning, spherical influence expansion,
interpolation, output cap, or LiDAR-selected active-coordinate generation is
used. A generated site can lack a radar return. The full 8³ child region of
each active coarse cell is generated; high confidence is learned separately.

Clean LiDAR labels are positive where a point was measured, negative only on
conservatively verified pre-return free space, and ignored elsewhere. The
spatial objective is known-site weighted BCE with logits plus 0.5 soft Dice.
Total student loss is `distillation_weight × SmoothL1 + spatial_weight ×
(BCE + dice_weight × Dice)`. This does not force reconstruction of every clean
LiDAR hit outside the radar-generated candidate domain. The student output is
confidence, not a final reconstructed point cloud.

The conventional radar-supported IoU uses predicted and clean occupied voxels
within the complete radar-generated candidate domain. The separately named
`radar_supported_iou_known` masks that domain to positive and verified free
voxels. The known-space value is the default checkpoint metric because unknown
space has no trustworthy negative label. Both, the selected threshold, and
candidate coverage (represented clean points / all clean points inside the
declared fixed PFE grid) are logged. Thresholds 0.1–0.9 are selected using
validation only; the exported logits and probabilities remain unthresholded.

## Data and dependencies

The VoD loader needs the public dataset's `lidar`, `radar_20frames_verified_doppler_radial`,
calibrations, `ImageSets`, and `label_2` for teacher object metrics. Radar is
calibrated into LiDAR coordinates before voxelization. Training uses the VoD
train split and online paired XY flip, translation, yaw, and scale; validation
is unchanged. The Stage I augmentation uses the same transform distribution
as the earlier BEV augmentation. TensorBoard and PNG plots show train and val
total/component losses, validation correspondence and confidence metrics,
and overfitting trends. Full validation every five epochs determines checkpoint
selection; fixed first-frame quick validation runs in intervening epochs.

Python, CUDA, PyTorch, `spconv`, MinkowskiEngine, NumPy, SciPy, tqdm, and
matplotlib are needed for all three GPU phases. TensorBoard is optional when
the trainers use `--no-tensorboard`; `metrics.jsonl` and PNG plots are still
written. Install `tensorboard` into the same Python environment and omit that
flag to write event files. The original
PFE constructor requires CUDA even though the mathematical unit tests are
CPU-compatible. On the professor machine, the existing user-space OpenBLAS
runtime must be on `LD_LIBRARY_PATH` for MinkowskiEngine. The GPU decoder
smoke test should pass before Phase 2 training. The full generative candidate
set may consume substantial GPU memory; no points are secretly removed.

TJ4DRadSet is not silently treated as VoD. `PairedRadarLidarDataset` is the
adapter contract, and `tj4dradset_dataset()` fails explicitly until calibrated
sensor pairing, split, per-point radar time index, and oriented instance-box
readers are implemented.

## Commands on the professor machine

Run from this repository root after updating the source. Use an output volume
with sufficient free space; the original `/mnt/3D10B36523559581` volume was
full during an earlier experiment. These commands are intentionally separate
so each phase can be inspected before starting the next.

```bash
BASE=/mnt/3D10B36523559581/Gianluca
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
PY="$BASE/Sensor-Fusion/.venv_model_v2/bin/python"
OUT=/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/crossmodal_raw64_voxel_020_020_025
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8
mkdir -p "$OUT"

# Backend coordinate and backward check; no dataset or encoder weights needed.
"$PY" -m scripts.smoke_crossmodal_raw_decoder

# Phase 0: two independent original-PFE pretrains.
"$PY" -u -m models.radar_lidar_crossmodal_v1.pretrain_encoder \
  --vod-root "$VOD" --modality radar --output-root "$OUT/radar" \
  --online-augmentation-config configs/stage1_online_geometric_augmentation.json \
  --epochs 25 --batch-size 2 --num-workers 2 --device cuda --no-tensorboard
"$PY" -u -m models.radar_lidar_crossmodal_v1.pretrain_encoder \
  --vod-root "$VOD" --modality lidar --output-root "$OUT/lidar" \
  --online-augmentation-config configs/stage1_online_geometric_augmentation.json \
  --epochs 25 --batch-size 2 --num-workers 2 --device cuda --no-tensorboard

# Phase 1: frozen original encoders, trainable local teacher.
"$PY" -u -m models.radar_lidar_crossmodal_v1.train_teacher \
  --vod-root "$VOD" --radar-encoder "$OUT/radar/best_encoder.pth" \
  --lidar-encoder "$OUT/lidar/best_encoder.pth" --output-root "$OUT/teacher" \
  --config configs/crossmodal_teacher_raw64.json \
  --online-augmentation-config configs/stage1_online_geometric_augmentation.json \
  --epochs 25 --batch-size 2 --num-workers 2 --device cuda --no-tensorboard

# Phase 2: frozen teacher and radar encoder, radar-only student.
"$PY" -u -m models.radar_lidar_crossmodal_v1.train_student \
  --vod-root "$VOD" --teacher-checkpoint "$OUT/teacher/best_object_r1.ckpt" \
  --output-root "$OUT/student" --config configs/crossmodal_student_raw64.json \
  --online-augmentation-config configs/stage1_online_geometric_augmentation.json \
  --epochs 25 --batch-size 1 --num-workers 2 --device cuda --no-tensorboard
```

For a one-frame integration check, run each trainer with a new smoke output
directory plus `--epochs 1 --train-limit 1 --val-limit 1 --quick-val-frames 1
--full-val-every 1 --num-workers 0`. Use smoke encoder checkpoints for a teacher
smoke, and the smoke teacher checkpoint for a student smoke. Do not interpret
one-frame scores as model quality. Resume a full run using its matching
`--resume "$OUT/<phase>/last.ckpt"` and original arguments.

Evaluate and export after training:

```bash
"$PY" -m models.radar_lidar_crossmodal_v1.evaluate teacher \
  --vod-root "$VOD" --checkpoint "$OUT/teacher/best_object_r1.ckpt" \
  --output "$OUT/teacher/final_val.json"
"$PY" -m models.radar_lidar_crossmodal_v1.evaluate student \
  --vod-root "$VOD" --checkpoint "$OUT/student/best_radar_iou.ckpt" \
  --output "$OUT/student/final_val.json"
"$PY" -m models.radar_lidar_crossmodal_v1.export_student \
  --vod-root "$VOD" --checkpoint "$OUT/student/best_radar_iou.ckpt" \
  --split val --frame-id 08236 --output-root "$OUT/student/raw"
"$PY" -m models.radar_lidar_crossmodal_v1.parameter_report \
  --teacher-checkpoint "$OUT/teacher/best_object_r1.ckpt" \
  --student-checkpoint "$OUT/student/best_radar_iou.ckpt"
```

Each run writes `metrics.jsonl`, plots, `last.ckpt`, and
its full-validation selected checkpoints. The student export is a `.pt` dict
with `conditioning_features` `[N,64]`, `conditioning_coords` `[N,4]`,
`confidence_logits` and `confidence_probabilities` `[M,1]`, `confidence_coords`
`[M,4]`, `grid_metadata`, and `interface_version`. Coordinates are
`[batch,z,y,x]`; conditioning coordinates are stride-8 cell indices, while
confidence coordinates are stride-1 fine indices. TensorBoard events are written
only when that optional dependency is installed and logging is enabled. The output is raw and ready
for future Stage II conditioning. Existing Stage II diffusion checkpoints are
**not** compatible with this new 64D/stride-8 interface without a new adapter.

The Windows workspace has no trained exact-PFE checkpoint or sparse CUDA
backend. Before this voxel-size change, on the professor's RTX 4090, one-frame GPU smoke tests passed for
both independent encoder pretrains, teacher training, student training,
checkpoint reload/evaluation, and raw radar-only export. The sparse decoder
expanded one stride-8 site to 512 fine sites and backpropagated finite
gradients. These runs verify integration only. Full training and held-out
quality assessment remain necessary. The one-frame smoke student represented
11.7% of clean returns inside the previous fixed PFE grid; that candidate-coverage
ceiling cannot improve through confidence training alone for that frame. Rerun
the GPU smoke test and all training phases with the new voxel size.
