# Hierarchical Stage I: local pilot, 31 October–1 November 2026

## Question and decision

Can a learned coarse-to-fine coordinate generator offer Stage II a sparse
candidate set with at least 90% recall of **missing clean LiDAR voxels** and
at least 20% precision? On this local pilot, **the generated pool can exceed
90% recall, but the selected sparse gate cannot**. This is a reason to run a
proper full-data experiment, not a reason to insert the current model into
Stage II. The old 64-slot checkpoint is absent here, so there is no matched
old-versus-new result yet.

## Setup

- VoD official training/validation partitions; 500 randomly sampled training
  frames out of 5,139, seed 3103. Initial model trained 10 epochs. The two
  short fine-tunes each started from the 10-epoch checkpoint and used the
  same 500 frames for 3 epochs.
- Inputs: 20-scan verified Doppler radar. Faults: complete LiDAR loss and
  severity-5 FOV filtering. The clean scan supplies missing-cell targets only
  during training and evaluation. At inference, the model sees radar and the
  surviving faulty scan.
- Metric: exact unique missing clean cells in Stage II's 409,600-cell region,
  `[0,32] × [-16,16] × [-2,2]` m, with 0.2 × 0.2 × 0.25 m cells. Precision is
  true missing cells divided by selected sites; recall is true missing cells
  divided by all missing target cells. This is stricter and differently scoped
  than the old 0.2 m-tolerant, radar-local scan-matching result.
- The first 64 validation frames were used for checkpoint selection. A separate
  random 128 validation frames were split 64/64 for site-budget calibration
  and held-out evaluation, excluding those 64 selection frames. The same 128
  frame IDs and fault modes were used for every row below. Results are pooled
  over cells, not averaged per frame.
- The pilot checkpoint-selection score could prefer a higher-recall site
  budget below 20% precision. This was corrected in the training script after
  these runs started. The fixed-budget held-out tables below do not use that
  score, but future checkpoints should be selected with the corrected rule.

## Held-out result (64 frames per arm)

| Weights and inference growth | Generated sites/frame | Total-loss pool recall | Total-loss P/R at 10k | FOV pool recall | FOV P/R at 10k |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 epochs, 2,400 coarse / 7,000 middle | 56,000 | 84.5% | 23.35% / 48.78% | 86.9% | 23.09% / 51.95% |
| Same weights, 3,200 / 20,000 | 160,000 | 96.3% | 23.35% / 48.78% | 97.1% | 23.09% / 51.95% |
| 3 more epochs with wider growth | 160,000 | 97.0% | 23.18% / 48.43% | 97.7% | 23.02% / 51.81% |
| 3 more epochs with hard-negative fine loss | 160,000 | 97.1% | 23.34% / 48.76% | 97.7% | 23.13% / 52.06% |

For the 10-epoch weights with wider growth, a 20,000-site gate reached 64.5%
recall at 15.4% precision for total loss, and 67.7% at 15.0% for FOV. Thus
simply raising the final site budget does not meet both requested constraints.
There were about 4,786 missing total-loss target cells per held-out frame:
90% recall at 10,000 selected sites would require roughly 43% precision;
at 20,000 selected sites it would require roughly 21.5% precision.
The 10,000-site gate is only 2.4% of the full region; its intermediate
160,000-site pool is 39.1% of the region and is **not** what Stage II would
activate. The pool's recall is an oracle upper bound: it assumes perfect
ranking among its generated sites.

The 1.6 m geometric radar-anchor reach diagnostic was 88.9% on total-loss
targets and 89.5% on FOV-missing targets in these same 64 held-out frames.
That explains why increasing old pointer count within the same radius cannot
guarantee 90% exact full-region recall. It is a geometric diagnostic, not a
measured result of the unavailable 64-slot checkpoint.

## Temporal-dependence check

The 64-frame held-out sample is **not independent by radar history**: 54 of
its 64 frames are within 20 frame IDs of a checkpoint-selection frame, and 55
are within 20 IDs of a calibration frame. The 20-scan radar stacks can
therefore overlap. An additional evaluation enforced a gap of more than 20
frame IDs from selection frames and among sampled frames. The greedy sample
yielded ten frames, split five calibration / five held-out.

| Gate on five time-separated held-out frames | Total-loss pool | Total-loss P/R at 10k | FOV pool | FOV P/R at 10k |
| --- | ---: | ---: | ---: | ---: |
| 10-epoch weights, original growth | 81.9% | 22.20% / 48.90% | 85.3% | 21.79% / 52.28% |
| Same weights, wider growth | 94.2% | 22.20% / 48.90% | 95.5% | 21.79% / 52.28% |
| Hard-negative fine-tune, wider growth | 94.9% | 22.32% / 49.18% | 96.0% | 22.07% / 52.95% |

The time-separated check supports the qualitative diagnosis but is much too
small for a reliable performance claim. The full run now defaults to choosing
checkpoint-selection frames in contiguous blocks across available frame-ID runs;
this leaves more independent frames for a later `--temporal-gap 20` comparison.

## What worked and what did not

Growth is useful: with the same trained weights, increasing coarse/middle
parents raised pool recall by about 12 percentage points on total loss.
Final ranking remains the bottleneck. Fine-only ranking on expanded parents
was unstable; the implementation now combines coarse, middle, and fine score
evidence. That fixes the ablation's score collapse, but it leaves the first
10,000 selected cells almost unchanged. Fine-tuning on the wider pool and
penalizing the top 10,000 false candidates did not materially improve the
held-out sparse gate in three local epochs.

The target is a thin measured LiDAR scan. A cell with no clean return is not
necessarily physical free space, so the negative loss should be interpreted
as scan-matching supervision, not occupancy truth. Radar-only inference under
complete LiDAR loss may not contain enough information to identify the exact
sampling pattern of a future clean scan at 0.2 m resolution. That is a
hypothesis to test with the full-data run and a less sample-specific surface
metric, not a conclusion established by this pilot.

## Required next experiment

Run the expanded configuration on all 5,139 training frames for the planned
25 epochs, use block validation selection, and compare its checkpoint with
the completed 64-slot checkpoint on identical time-separated frames and exact
missing-cell targets. The commands are in `HIERARCHICAL_STAGE1_TRIAL.md`.
Report the 10k, 15k, 20k, and 30k gates, runtime/memory, and candidate-pool
recall. Do not retrain Stage II against this pilot checkpoint: its feature
distribution differs from the existing Stage I encoder, and its gate must be
cached by fault mode. If the full run still stalls near 50% recall at 20%
precision, reconsider the exact-scan coordinate objective or generate support
inside Stage II with a coarse sparse decoder and observed-LiDAR context.

The local checkpoint and JSON comparisons are saved under
`C:\Users\gianl\Desktop\Thesis\hierarchical_stage1_runs_2026-11-01`.
The 10-epoch reference checkpoint is
`hierarchical_stage1_trial_500x10_20261031/best.ckpt` (SHA-256 prefix
`df73c2a6bd0d0882`). The JSON files include full counts, site budgets,
fault-specific calibration/held-out scores, and checkpoint hashes.
