# Native Stage-I sparse encoders, cross-modal teacher, radar-only student

## Joint relationship and radar-only occupancy experiment

`train_meta_occupancy.py` implements the proposed bilevel experiment without
changing the older teacher/student training scripts. Start with independently
pretrained radar and clean-LiDAR encoders:

```powershell
python -m models.radar_lidar_crossmodal_v1.train_meta_occupancy `
  --vod-root PATH_TO_VOD --radar-encoder PATH_TO_RADAR_ENCODER `
  --lidar-encoder PATH_TO_LIDAR_ENCODER --output-root PATH_TO_NEW_RUN
```

For fixed, unaugmented VoD frames, build the meta cache once per split and
reuse it for every epoch. The builder reads the existing fixed-frame cache.
It stores frozen radar and LiDAR bottleneck features, exact 3.2 m neighbors,
radar-only candidate coordinates, clean occupancy and known-free labels,
radar support, and supported clean voxel coordinates for scan metrics. It
never stores trainable teacher attention, distilled student features, decoder
logits, or probabilities. An interrupted build resumes missing frames.

```bash
for split in train val; do
  python -m scripts.cache_meta_occupancy \
    --vod-root "$VOD" --data-cache-root "$PREV/fixed_frames" \
    --cache-root "$OUT/meta_cache" \
    --radar-encoder "$PREV/full/radar/best_encoder.pth" \
    --lidar-encoder "$PREV/full/lidar/best_encoder.pth" \
    --split "$split" --num-workers 2 --device cuda
done

python -m models.radar_lidar_crossmodal_v1.train_meta_occupancy \
  --vod-root "$VOD" --meta-cache-root "$OUT/meta_cache" \
  --radar-encoder "$PREV/full/radar/best_encoder.pth" \
  --lidar-encoder "$PREV/full/lidar/best_encoder.pth" \
  --output-root "$OUT/train" --epochs 25 --warmup-epochs 2 \
  --batch-size 8 --num-workers 2 --scan-metrics-every 5 \
  --quick-val-frames 32 --full-val-every 5 --device cuda
```

The manifest binds this cache to the VoD frame IDs, encoder hashes, grid,
radius, and geometry/label source code. Every cached candidate-coordinate
tensor is checked against the live decoder output. Validation uses 32 frames
on ordinary epochs and the full split every five epochs (and at the final
epoch); checkpoint selection uses full validation only. The four scan metrics
remain full-split measurements every five epochs. The old student supervision
cache has a different source fingerprint and is not silently accepted.
Before the full build, create two training entries with `--split train --limit 2`
and compare them to the uncached path with `python -m scripts.smoke_meta_cache`
using the same root and encoder arguments. The smoke checks student features,
decoder logits, occupancy masks, support, and teacher features.

The default config is `configs/crossmodal_meta_occupancy.json`. Teacher
attention takes radar queries and up to 24 clean-LiDAR bottleneck candidates
within **3.2 m** (Euclidean distance in the calibrated LiDAR frame). A
candidate farther away is excluded even if fewer than 24 remain. The student
starts with two map-only warmup epochs. On subsequent epochs, each teacher
update uses a paired training batch A to compute its LiDAR-conditioned feature,
simulates one differentiable student feature-distillation step, and scores the
**radar-only** student map on a distinct training batch B. The quiz BCE flows
back through that simulated step to the teacher attention and gate. Small
geometric-attention and LiDAR-feature anchors keep the relationship branch
grounded; their weights are configurable. A real student step on A then uses
known-voxel map BCE plus Smooth L1 distillation from the updated teacher.
The decoder consumes the student's distilled feature without direct pyramid
skip connections, and no teacher or LiDAR tensor enters its forward path.
`--teacher-init` optionally initializes the
teacher from an existing compatible teacher checkpoint; its weights are then
trained with the new 3.2 m neighborhood. `--resume` restores a joint run.

Fine voxel labels use the existing 0.2 × 0.2 × 0.25 m grid. A generated
candidate is scored only when its center is at most 3.2 m from a radar point
in the same frame **and** clean LiDAR marks it occupied or conservatively
known free. Unobserved voxels do not become negatives. BCE is unweighted so
its sigmoid can be assessed as a probability; validation logs Brier score,
known-space IoU at a fixed 0.5 threshold, and the fraction of clean returns
inside radar support represented by generated, supported candidates.
The present stride-8 generative decoder emits the 8³ children of its active
radar cells. It does **not** emit every voxel in the 3.2 m union of spheres.
The reported `iou_known_supported` is on the intersection of that sphere
support and generated candidates. `iou_full_support` additionally counts every
nearby clean occupied voxel absent from the decoder output as a false negative;
this is the checkpoint-selection metric. Coverage reports the missing support.

Starting at epoch 5, validation also logs `iou_exact`, `iou_at_0_2m`,
`f1_exact`, and `f1_at_0_2m` every five epochs (change with
`--scan-metrics-every`). These compare thresholded predicted occupied voxels
with **all** clean-LiDAR occupied voxels whose centers are in the 3.2 m radar
support, including voxels the decoder never generated. Exact means the same
voxel key. The 0.2 m metrics use maximum one-to-one matching of voxel centers
within 0.2 m and never match across frames. For each metric,
`IoU = TP / (predicted + observed - TP)` and
`F1 = 2 TP / (predicted + observed)`. Predictions in LiDAR-unobserved space
count as unmatched predictions here, so these are clean-scan agreement metrics,
not definitive physical occupancy scores. Existing BCE, Brier, and known-space
IoU continue to be evaluated every epoch.

`last.ckpt` and `best_iou.ckpt` contain both training models. Their matching
`last_student.ckpt` and `best_student.ckpt` files contain only the radar-only
inference model:

```python
from models.radar_lidar_crossmodal_v1.meta_occupancy import radar_only_probability_map
from models.radar_lidar_crossmodal_v1.train_meta_occupancy import load_meta_student

student, checkpoint = load_meta_student("PATH_TO_RUN/best_student.ckpt")
output = radar_only_probability_map(
    student, radar_points, radar_valid, checkpoint["radius_m"])
```

The virtual update uses a one-step SGD proxy for the student's feature-path
update; the real student optimizer remains AdamW. It needs PyTorch 2.3+ and
appreciably more GPU memory than ordinary student training. The full run also
requires the existing native sparse encoder and MinkowskiEngine installation.

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
The raw student's stride-8 features expand through three non-overlapping
stride-2 generative layers. Thresholded scores can therefore appear in
rectangular patches; the older five-panel trilinear and edge-aware views were
interpolated from a different Stage-I proposal head.

Cross-modal training now crops clean LiDAR into a fixed forward radar sector
(±70° azimuth, 80 m), using each frame's radar-to-LiDAR calibration. Radar
input remains unchanged and student inference still takes radar alone. The
crop precedes encoder and teacher training, student labels, and cache writes;
the viewer uses the same sector. Existing LiDAR, teacher, and student weights
and LiDAR-dependent caches from the 360° setup must be rebuilt in a new run.
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
  --epochs 25 --batch-size 16 --num-workers 2 --device cuda --no-tensorboard
"$PY" -u -m models.radar_lidar_crossmodal_v1.pretrain_encoder \
  --vod-root "$VOD" --modality lidar --output-root "$OUT/lidar" \
  --online-augmentation-config configs/stage1_online_geometric_augmentation.json \
  --epochs 25 --batch-size 16 --num-workers 2 --device cuda --no-tensorboard

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

For a completed teacher, `bash scripts/train_crossmodal_student_cached.sh`
builds a two-frame validation cache, checks cached training/validation in the
foreground, completes the fixed validation cache, then resumes or starts only
the radar-only student. It uses validation batches of two and computes all
threshold IoUs on the GPU before transferring aggregate counts. Cached entries
include exact candidate coordinates and a teacher/source fingerprint; mismatched
coordinates or stale teacher weights stop validation. Training remains online
because geometric augmentation changes its coordinates each epoch. The script
uses a tiled GPU teacher-neighbor query during cache construction and training;
set `CROSSMODAL_GPU_NEIGHBORS=0` to use the original CPU KD-tree if the GPU
query is slower on a particular machine.

For the next student trial, `bash scripts/train_crossmodal_student_noaug_cached.sh`
disables geometric augmentation, builds separate fixed train and validation
supervision caches, checks both with a two-frame foreground smoke run, and then
trains a fresh 25-epoch student in `full/student_noaug_cached`. Each entry stores
frozen teacher features, correspondence evidence, and clean-LiDAR supervision
on the radar-only candidate coordinates. The train and validation caches are
checked against the teacher hash, source version, split, and generated
coordinates. The cache builder estimates disk use after eight new frames and
stops if the remaining space is insufficient. This trial still uses the
already-trained encoder and teacher checkpoints; it retrains only the student.

For a complete from-scratch no-augmentation trial, run
`bash scripts/train_crossmodal_noaug_30ep.sh` in the foreground. This uses a new
output root and trains the radar encoder, LiDAR encoder, cross-attention teacher,
and radar-only student for 30 epochs each. It first caches aligned radar and
clean LiDAR for all train and validation frames. Once the encoders exist, it
caches their frozen bottleneck features, fixed neighborhoods, and object labels
for teacher training. Once the teacher exists, it caches its features,
correspondence evidence, and clean-LiDAR candidate labels for student training.
Each phase's two-frame cache and smoke test runs before its full training. The
teacher and student caches cannot be constructed before their upstream weights
exist; the script constructs them immediately after those weights finish.
The new default output is `crossmodal_noaug_30ep_20261015` and never resumes
checkpoints from augmentation-trained runs. Partial caches and matching new-run
checkpoints are reusable if the foreground script is restarted.

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

## Lossless teacher cache conversion

Teacher training accepts both the existing `.pt` static cache and a new
versioned `.ptz` format. A new teacher cache directory uses compact storage by
default; an existing legacy directory stays readable. The compact format
delta-packs stride-8 coordinates, stores neighbor indices in the smallest
lossless integer width, byte-shuffles feature and distance tensors, then
compresses them with zlib level 9 by default. Dtypes and every tensor bit are
restored exactly. Optional zstd requires `zstandard` in the Python environment.

Convert an existing cache without recomputing its frozen encoders or changing
the source files. The first command samples 50 frames per split and prints
the measured size ratio; the second resumes and converts the complete cache:

```bash
OLD=/path/to/existing/teacher_static
NEW=/path/to/teacher_static_compact
"$PY" -u -m scripts.compact_crossmodal_teacher_cache \
  --source-root "$OLD" --destination-root "$NEW" --limit 50
"$PY" -u -m scripts.compact_crossmodal_teacher_cache \
  --source-root "$OLD" --destination-root "$NEW"
```

Each destination entry is decoded and compared bit-for-bit with the source
before it is accepted. The converter is resumable and never removes the old
cache. Use `--static-cache-root "$NEW"` for teacher training; the trainer
selects the matching reader automatically. For a future complete no-augmentation
run, set `TEACHER_CACHE="$NEW"` before invoking
`scripts/train_crossmodal_noaug_30ep.sh`. This cache supports teacher training.

## Radar-only surface proposal trial

The earlier October 13 view used learned surface proposals before optional
interpolation. Train the cross-attention teacher, then the **feature-only**
radar student with `train_feature_student`, then the confidence and surface
decoder with `train_surface_student`. Feature distillation does not train a
confidence decoder. The surface decoder loads and freezes the trained
student's radar encoder and 64-channel distilled bottleneck. Its
query comes from that bottleneck at each fine radar anchor; fine radar
pyramid features provide local detail.
The first trial grouped fine radar sites at stride 8 and predicted two
locations per anchor. That could produce only 982 raw proposals from 491
anchors even with the display threshold at zero. The pointer configuration
uses every active stride-1 radar voxel as an anchor and predicts up to 24
continuous locations with strengths. It merges pointers into unique fine
voxels; no neighboring output cells or spherical region are filled. Clean
LiDAR supervises only proposed voxels where occupancy or free space is known.
The teacher is absent from this phase, and inference uses radar alone. Older
surface checkpoints remain loadable with their original per-pointer viewer.
The new loss and 24-slot head require a new decoder checkpoint.

The existing fixed student supervision cache contains frozen teacher
bottleneck features, which `train_feature_student` can reuse without calling
the teacher each batch. Its fixed-grid occupancy labels cannot supervise the
moving pointers; clean occupancy and visibility are queried at their newly
proposed coordinates.
The fixed aligned radar/LiDAR frame cache may still be supplied with
`--data-cache-root`. Use new output directories for both stages; the trained
feature-only student checkpoint is required and must use the same
radar variant and forward LiDAR field of view.

Run a one-frame smoke test in the foreground before the full run:

```bash
PY=/mnt/3D10B36523559581/Gianluca/Sensor-Fusion/.venv_model_v2/bin/python
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
TEACHER=/path/to/teacher/best_object_r1.ckpt
TRAIN_CACHE=/path/to/student_supervision_compact/train
VAL_CACHE=/path/to/student_supervision_compact/val
FEATURE_RUN=/path/to/new/feature_student_run
SURFACE_RUN=/path/to/new/surface_student_run
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/mnt/3D10B36523559581/Gianluca/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8

"$PY" -u -m models.radar_lidar_crossmodal_v1.train_feature_student \
  --vod-root "$VOD" --teacher-checkpoint "$TEACHER" \
  --training-cache-root "$TRAIN_CACHE" --validation-cache-root "$VAL_CACHE" \
  --output-root "${FEATURE_RUN}_smoke" \
  --epochs 1 --batch-size 1 --validation-batch-size 1 \
  --train-limit 1 --val-limit 1 --quick-val-frames 1 \
  --full-val-every 1 --num-workers 0 --device cuda

"$PY" -u -m models.radar_lidar_crossmodal_v1.train_feature_student \
  --vod-root "$VOD" --teacher-checkpoint "$TEACHER" \
  --training-cache-root "$TRAIN_CACHE" --validation-cache-root "$VAL_CACHE" \
  --output-root "$FEATURE_RUN" \
  --epochs 30 --batch-size 8 --validation-batch-size 8 \
  --full-val-every 5 --num-workers 4 --device cuda

"$PY" -u -m models.radar_lidar_crossmodal_v1.train_surface_student \
  --vod-root "$VOD" --feature-student-checkpoint "$FEATURE_RUN/best_distillation.ckpt" \
  --output-root "${SURFACE_RUN}_smoke" \
  --config configs/crossmodal_surface_student.json \
  --epochs 1 --batch-size 1 --validation-batch-size 1 \
  --train-limit 1 --val-limit 1 --quick-val-frames 1 \
  --full-val-every 1 --num-workers 0 --device cuda

"$PY" -u -m models.radar_lidar_crossmodal_v1.train_surface_student \
  --vod-root "$VOD" --feature-student-checkpoint "$FEATURE_RUN/best_distillation.ckpt" \
  --output-root "$SURFACE_RUN" --config configs/crossmodal_surface_student.json \
  --epochs 30 --batch-size 4 --validation-batch-size 4 \
  --full-val-every 5 --num-workers 4 --device cuda

"$PY" -m scripts.compare_crossmodal_surface \
  --vod-root "$VOD" --checkpoint "$SURFACE_RUN/best_proposal_utility.ckpt" \
  --split val --frame-id 08236 00405 \
  --output-root "$SURFACE_RUN/views" --device cuda
```

The decoder predicts up to 24 optional fine-voxel pointers per active radar
voxel. Pointers from one radar voxel to the same destination count once;
pointers from different radar voxels raise that destination's bounded
confidence. Training compares only pointed voxels with clean-LiDAR occupancy
or verified free space, using masked BCE, Dice, and a distinct-hit reward.
Unknown voxels are ignored, and unproposed clean voxels are not penalized.
The checkpoint metric is distinct correct proposed voxels times precision on
verified proposals, aggregated over the full validation split.
If a trained sequential four-pointer checkpoint used the exact same distilled
feature student, add `--initialize-from-surface-checkpoint /path/to/old.ckpt`
to the new decoder run. This copies its first four pointer slots and keeps the
remaining 20 newly initialized. A full run with no occupied pointed voxels in
its first epoch stops with a clear error because the proposal-only objective
has no positive signal. A tiny smoke run prints a warning instead.

The viewer writes uncapped merged raw-voxel and edge-aware point clouds, a
synchronized four-panel HTML view with radar and forward-sector LiDAR as
references, and per-frame comparison metrics. Its display threshold starts at
zero, so all raw candidate voxels are visible until the slider is moved. The
edge-aware field is derived
only from predicted positions, scores, and radar features. It is never fed
back to the student or used as a training label. These new raw proposals are
an alternative Stage I interface; the current Stage II candidate loader still
expects the older fixed-grid confidence format and needs adaptation before it
can consume this checkpoint.
