# Hierarchical Stage I coordinate trial

This is an isolated trial of a learned sparse coordinate generator for Stage II.
It does **not** replace the 64-slot Stage I checkpoint or retrain Stage II.
The comparison target is exact **missing clean-LiDAR voxels** in Stage II's
`[0,32] × [-16,16] × [-2,2]` m region, with 0.2 × 0.2 × 0.25 m fine cells.
This is a different metric from the older 1 m radar-local, 0.2 m-tolerant scan
match. `scripts/compare_hierarchical_stage1.py` can score both checkpoint types
on the *same* missing-cell target and validation frames.

## Architecture

`hierarchical_gate.py` encodes radar and optional faulty LiDAR separately as
per-voxel point summaries. A small coarse 3D CNN processes all 4 × 40 × 40
cells at 0.8 × 0.8 × 1 m. It scores coarse parents, expands them to eight
children at 0.4 × 0.4 × 0.5 m, scores/prunes again, and scores fine children.
The intermediate budgets limit growth; final top-K limits Stage II's active
sites. No clean LiDAR is used to build inference features or select sites.
During training only, positive target parents are retained so a prematurely
pruned parent cannot remove all fine supervision. Losses at all three scales
favor positive clean cells while penalizing cells with no return in the target
scan. That negative means *no measured return*, not proven physical free
space. The model can
run with empty faulty LiDAR for complete sensor loss.
Final candidates are ranked by the geometric mean of coarse, middle, and
fine-stage scores. That score is **not calibrated**; top-K budgets are the
primary comparison. `--score-mode fine_only` is an ablation that ignores the
parent evidence and can fail when intermediate budgets are increased.

The fine ROI contains 409,600 possible cells. A 20,000-site gate occupies
4.9% of it; a 40,000-site gate occupies 9.8%. For a frame with 5,000 missing
target cells, 90% recall at 20% precision requires about 22,500 selected
sites. Thus 40,000 sites could raise recall while failing the precision goal
even with perfect coverage. Report both the exact hit rate and actual selected
site count rather than calling a larger gate an improvement by recall alone.

This adapts the learned support growth and pruning of [GSDN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490290.pdf),
the coarse-to-fine sparse occupancy decoding of [SparseOcc](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03570.pdf),
and the multi-scale supervision used in [SG-NN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dai_SG-NN_Sparse_Generative_Neural_Networks_for_Self-Supervised_Scene_Completion_of_CVPR_2020_paper.pdf).
None of those papers establishes the target 90% recall / 20% precision for
VoD radar; it remains an experimental criterion.

## Full trial on the professor machine

The professor-machine workflow used for earlier thesis runs is a pinned
GitHub checkout on the data drive. After the trial commit has been pushed,
set `REV` to that full commit hash and run:

```bash
SRC=/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-hierarchical-stage1-20261101
REV=REPLACE_WITH_FULL_PUBLISHED_COMMIT_HASH
if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/Gianluca3103/Thesis---1.git "$SRC"
fi
git -C "$SRC" fetch origin main
git -C "$SRC" checkout --detach "$REV"
cd "$SRC"
bash scripts/run_hierarchical_stage1_full.sh
```

It checks the existing fixed-frame cache and CUDA environment, makes a
one-frame smoke checkpoint, trains for 25 epochs, and compares the new gate
with the 64-slot checkpoint if that checkpoint is present. `REPO`, `DATA`,
`VOD`, `PY`, `RAW`, `CONFIG`, `OUT`, `BASELINE`, and `WORKERS` can be supplied
as environment variables when the professor-machine paths differ. Run it in
`tmux` if the SSH connection may close.

Use a new output directory. The expanded configuration is the recommended
full-run starting point because the local pilot's default 56,000-candidate
pool did not reach 90% target recall. `DATA_CACHE_ROOT` is optional, but should point
to the same fixed-frame cache used by the 64-slot experiment when available.
The default input is the same 20-scan radar variant. The two default faults
are complete loss and severity-5 FOV filtering; the model rotates through them
on training frames and evaluates both on validation frames. Add `fog_sim` plus
`--fog-cache-root` only when the compatible severe fog cache exists.

```bash
REPO=/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-hierarchical-stage1-20261101
DATA=/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
PY=/mnt/3D10B36523559581/Gianluca/Sensor-Fusion/.venv_model_v2/bin/python
OUT="$DATA/hierarchical_stage1_v1"
BASELINE="$DATA/fine_association_1p6m64_20261031/stage1/best.ckpt"

cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
"$PY" -u -m scripts.train_hierarchical_stage1 \
  --vod-root "$VOD" --output-root "$OUT" \
  --config configs/hierarchical_stage1_gate_expanded_v1.json \
  --data-cache-root "$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames" \
  --epochs 25 --val-limit 128 --val-selection blocks --device cuda

"$PY" -u -m scripts.compare_hierarchical_stage1 \
  --vod-root "$VOD" --checkpoint "$OUT/best.ckpt" \
  --baseline-checkpoint "$BASELINE" \
  --limit 256 --temporal-gap 20 \
  --budgets 5000 10000 15000 20000 30000 \
  --output "$OUT/matched_heldout.json" --device cuda
```

The comparison excludes the frames used to choose the new model checkpoint.
With `--temporal-gap 20`, it also excludes frames near those frames and keeps
the sampled calibration and held-out frames at least 21 frame IDs apart. The
reported sample count may therefore be smaller than `--limit`. It splits the
remaining sample into budget-calibration and held-out halves and reports
exact candidate recall, precision, pool-oracle recall,
and active-site counts for each fault and model. The total-loss arm is the
cleanest architectural comparison because neither model sees faulty LiDAR.
The FOV arm additionally tests whether using surviving faulty returns for
coordinate generation helps.

The local prototype and its limitations are recorded in
`HIERARCHICAL_STAGE1_LOCAL_RESULTS.md`. Those numbers use 500 of the 5,139
training frames and do not substitute for this full run. The old 64-slot
checkpoint was not available locally, so its matched comparison must run
where that checkpoint and MinkowskiEngine are installed.

If the learned candidate pool misses too many targets, rerun the comparison
with larger `--coarse-parent-budget` and/or `--middle-parent-budget`. These
are inference-only growth ablations using the same trained weights; the
reported active-site count and precision must be checked again. They do not
repair a poor final-cell ranking.

## Decision criterion

Treat the trial as promising only if held-out **exact** missing-cell recall
improves materially over the 64-slot baseline at comparable active-site count
and reaches at least 20% candidate precision. The long-term target is at least
90% recall at that precision with acceptable Stage II memory and runtime.
The `pool` row is an upper bound for the *trained model's generated candidate
pool*. If pool recall stays below 90%, final score or threshold tuning cannot
reach the target; change parent growth/budgets or the feature model first.

The exported map has the Stage II voxel-coordinate and 64-channel feature
shape contract, but current Stage II training caches are tied to their old
Stage I checkpoint. End-to-end Stage II use requires a fresh gate cache and
retraining after this trial meets the coordinate criterion. Fault-dependent
gates also require cache generation per fault rather than the present fixed
radar-only gate cache.

## Interactive sample views

`scripts.visualize_hierarchical_stage1` writes an `index.html` with the same
rotatable point-cloud panels as earlier Stage I viewers. It shows radar,
clean LiDAR, the selected gate, missing clean target cells, surviving faulty
LiDAR, and an exact TP/FP/FN audit. Gate points are **voxel centers available
to Stage II**, not reconstructed LiDAR returns. Each frame/fault also gets
gate and missing-target PLY files plus counts in `summary.json`.

While training is running, use CPU inference in a second terminal so the
viewer does not take training GPU memory. The checkpoint is saved atomically:

```bash
SRC=/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-hierarchical-stage1-20261101
DATA=/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs
OUT="$DATA/hierarchical_stage1_expanded_full_20261101"
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
PY=/mnt/3D10B36523559581/Gianluca/Sensor-Fusion/.venv_model_v2/bin/python
cd "$SRC"
export PYTHONPATH="$SRC"
"$PY" -u -m scripts.visualize_hierarchical_stage1 \
  --vod-root "$VOD" --checkpoint "$OUT/last.ckpt" \
  --output-root "$OUT/views_current_top10000" \
  --limit 5 --temporal-gap 20 --budget 10000 --device cpu
```

After training and matched comparison complete, use the selected checkpoint
and the comparison's actual held-out frame IDs:

```bash
"$PY" -u -m scripts.visualize_hierarchical_stage1 \
  --vod-root "$VOD" --checkpoint "$OUT/best.ckpt" \
  --comparison "$OUT/matched_heldout.json" \
  --output-root "$OUT/views_best_heldout_top10000" \
  --limit 5 --budget 10000 --device cuda
```

Open the printed `index.html` in a browser. For a chosen validation frame,
`--frame-id 08236 00405` can replace automatic sampling; manually chosen
frames are qualitative examples and may have influenced checkpoint selection.
