# Native Stage-I sparse encoders, cross-modal teacher, radar-only student

This version is separate from `radar_lidar_stage1`, `radar_lidar_stage2`, and
their existing checkpoints. The active radar and LiDAR encoders use the native
Stage-I `PointVoxelEncoder` and `SparseBackbone` from
`models/radar_lidar_stage1/sparse.py`. Both have independent weights and
are retrained for this pipeline. The former Downloads encoder in `vendor/`
is retained for provenance but is no longer used by these training phases.

The earlier backbone has sparse scale widths 16, 32, 64 and 128, with axial
support growth at the fourth level. A trainable
128→64 adapter at its stride-8 output keeps the teacher and student at 64D,
with four 16D attention heads. The base voxel remains 0.2 × 0.2 × 0.25 m.
The earlier interpolation report records a Stage-I surface-region checkpoint
with those exact channels and growth setting. Its surface proposal head and
ellipsoid, trilinear, and edge-aware interpolation produced the displayed
five-panel views; this pipeline preserves its requested raw confidence output.
The current cross-modal grid has 512 × 512 × 16 fine cells and 64 × 64 × 2
stride-8 cells. The earlier Stage-I training run used a much wider grid;
its checkpoints cannot be loaded directly into this adapted encoder.

The radar point encoder embeds point offsets, global XYZ, RCS, radial and
compensated Doppler velocity, and scan age, then aggregates mean, maximum,
and point count per voxel. The LiDAR encoder uses the same geometry path plus
reflectivity. These are the earlier Stage-I
feature paths. Old exact-Downloads encoder, teacher, and student checkpoints
are incompatible with this version; start new runs for all three phases.

## Phases and parameters

| Phase | Frozen | Trained | Output |
| --- | --- | --- | --- |
| 0 radar | none | native Stage-I radar encoder, 128→64 adapter, temporary 512-bin head | `radar/best_encoder.pth` |
| 0 LiDAR | none | native Stage-I LiDAR encoder, 128→64 adapter, temporary 512-bin head | `lidar/best_encoder.pth` |
| 1 teacher | both pretrained sparse encoders | separate residual adapters, bottleneck Q/K/V/O, scalar gate, alignment projections | radar-coordinate LiDAR-aware 64D features |
| 2 student | radar sparse encoder and complete teacher | shared residual transform, feature MLP, three-level generative sparse confidence decoder | radar-only 64D features and raw fine-voxel confidence |

Phase 0 trains both encoders independently from scratch. Its pretraining
objective masks input features in complete stride-8 cells and predicts the
distribution of observed points over their 512 fine child cells. It does not
assert that unmeasured cells are empty and does not itself learn radar–LiDAR
correspondence. Sparse input coordinates remain present during masking, so this
is a feature-masking objective rather than a fully hidden geometry task. The
loss mask now selects the same stride-8 parent cells as the input mask. The
512-bin head is discarded. `best_encoder.pth` contains the point encoder,
sparse backbone, and 128→64 adapter and is loaded strictly by Phase 1.
The all-in-one runner trains the two encoders concurrently on the same GPU,
with a separate epoch bar and log for each. If either fails, it stops both;
rerunning resumes each encoder from its own `last.ckpt`.
Current-frame radar returns are used for
reliable object-label assignment; the full stack supplies encoder geometry and
Doppler velocity features.

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
radar sparse encoder feeds a trainable shared residual transform and two heads.
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
During student training, frozen teacher attention is averaged across four heads
and summed for LiDAR bottleneck neighbors sharing each fine candidate's stride-8
parent; the maximum evidence over radar queries increases only positive BCE
weights by `1 + lambda_teacher_evidence × evidence` (default λ = 0.5).
Known-free labels keep weight one, and LiDAR remains the occupancy target.
Setting λ to zero recovers the original positive-class-weighted BCE. Teacher
evidence does not create candidates or enter radar-only student inference.
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
declared fixed grid) are logged. Thresholds 0.1–0.9 are selected using
validation only; the exported logits and probabilities remain unthresholded.
The diagnostic `teacher_correspondence_supported_iou` also masks known sites
to candidates with teacher evidence at least `teacher_evidence_threshold`
(default 0.2). Its intersection and union are summed across validation frames.
It does not affect threshold or checkpoint selection; an empty region reports
`null` IoU.

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

Python, CUDA, PyTorch, MinkowskiEngine, NumPy, SciPy, tqdm, and
matplotlib are needed for all three GPU phases. TensorBoard is optional when
the trainers use `--no-tensorboard`; `metrics.jsonl` and PNG plots are still
written. Install `tensorboard` into the same Python environment and omit that
flag to write event files. Full training uses CUDA; small native sparse-encoder
checks run on CPU. On the professor machine, the existing user-space OpenBLAS
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
OUT=/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/crossmodal_native_sparse_verified_20261015_25ep
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8
mkdir -p "$OUT"

# Backend coordinate and backward check; no dataset or encoder weights needed.
"$PY" -m scripts.smoke_crossmodal_raw_decoder

# Phase 0: retrain the previous Stage-I radar and LiDAR encoder architecture.
"$PY" -u -m models.radar_lidar_crossmodal_v1.pretrain_encoder \
  --vod-root "$VOD" --modality radar --output-root "$OUT/radar" \
  --online-augmentation-config configs/stage1_online_geometric_augmentation.json \
  --epochs 25 --batch-size 4 --num-workers 2 --device cuda --no-tensorboard
"$PY" -u -m models.radar_lidar_crossmodal_v1.pretrain_encoder \
  --vod-root "$VOD" --modality lidar --output-root "$OUT/lidar" \
  --online-augmentation-config configs/stage1_online_geometric_augmentation.json \
  --epochs 25 --batch-size 4 --num-workers 2 --device cuda --no-tensorboard

# Phase 1: frozen pretrained encoders, trainable local teacher.
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

The Windows workspace has no trained checkpoint for this adapted encoder and
no sparse CUDA backend. Run a one-frame GPU smoke test of each phase on the
professor machine before starting full training. Previous one-frame results
from the Downloads-PFE version do not establish the quality of this encoder.
