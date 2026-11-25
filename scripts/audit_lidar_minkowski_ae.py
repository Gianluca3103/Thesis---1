"""Audit target leakage and clean-input copying in the Minkowski LiDAR AE."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.lidar_minkowski_ae import SparseVoxelAutoencoder, _inside, _keys
from models.lidar_minkowski_ae_data import CleanVoxelFrames, summarize_counts
from models.radar_lidar_stage2.sparse_geometry import GRID
from scripts.train_lidar_minkowski_ae import _counts


def remove_coarse_blocks(cells: np.ndarray, *, fraction: float, seed: int
                         ) -> tuple[np.ndarray, np.ndarray, int]:
    """Hide entire stride-2 blocks so no surviving voxel marks their sites."""
    if cells.ndim != 2 or cells.shape[1] != 3 or not len(cells):
        raise ValueError("Expected nonempty XYZ voxel coordinates")
    if not 0 < fraction < 1:
        raise ValueError("Block removal fraction must lie in (0,1)")
    coarse, inverse = np.unique(cells // 2, axis=0, return_inverse=True)
    if len(coarse) < 2:
        raise ValueError("Need at least two occupied coarse blocks")
    remove_count = min(len(coarse) - 1, max(1, round(len(coarse) * fraction)))
    chosen = np.random.default_rng(seed).choice(len(coarse), remove_count,
                                                 replace=False)
    removed = np.isin(inverse, chosen)
    return cells[~removed], cells[removed], remove_count


def _batched(cells: np.ndarray, device: torch.device) -> torch.Tensor:
    coords = np.column_stack((np.zeros(len(cells), dtype=np.int32), cells))
    return torch.as_tensor(coords, dtype=torch.int32, device=device)


def _selected(decoded, threshold: float) -> torch.Tensor:
    cutoff = math.log(threshold / (1 - threshold))
    return decoded.coordinates[decoded.logits >= cutoff]


def _candidate_signature(decoded) -> tuple[torch.Tensor, torch.Tensor]:
    valid = _inside(decoded.coordinates)
    keys = _keys(decoded.coordinates[valid])
    order = torch.argsort(keys)
    return keys[order], decoded.logits[valid][order]


def _add(into: dict[str, int], counts: dict[str, int]) -> None:
    for name, value in counts.items():
        into[name] += value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--removed-block-fraction", type=float, default=0.3)
    args = parser.parse_args()
    if args.frames < 1 or not 0 < args.threshold < 1:
        parser.error("Frames must be positive and threshold must lie in (0,1)")
    if not torch.cuda.is_available():
        parser.error("This audit needs the MinkowskiEngine CUDA environment")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("grid") != asdict(GRID):
        raise ValueError("Checkpoint grid differs from Stage II")
    model = SparseVoxelAutoencoder(**saved["model"]).cuda().eval()
    model.load_state_dict(saved["model_state"])
    if model.levels != 1:
        parser.error("This stride-2 block audit is for the one-level trial")
    data = CleanVoxelFrames(args.vod_root, "val", cache_root=args.cache_root,
                            limit=args.frames)
    totals = {name: {"tp": 0, "predicted": 0, "target": 0}
              for name in ("clean_to_clean", "zero_latent_features",
                           "block_masked_to_clean")}
    candidate_counts = {"tp": 0, "predicted": 0, "target": 0}
    removed_tp = removed_target = fine_sites = latent_sites = 0
    removed_blocks = 0
    with torch.no_grad():
        for index in tqdm(range(len(data)), desc="Minkowski latent audit", unit="frame"):
            row = data[index]
            fine = _batched(row["cells"], torch.device("cuda"))
            latent = model.encode(fine)
            decoded = model.decode(latent, fine_targets=None,
                                   threshold=args.threshold)
            diagnostic = model.decode(latent, fine_targets=fine,
                                      threshold=args.threshold)
            keys, scores = _candidate_signature(decoded)
            diagnostic_keys, diagnostic_scores = _candidate_signature(diagnostic)
            if (not torch.equal(keys, diagnostic_keys)
                    or not torch.allclose(scores, diagnostic_scores,
                                          rtol=1e-6, atol=1e-7)):
                raise RuntimeError(f"Targets changed decoder predictions for {row['frame_id']}")
            _add(candidate_counts, _counts(decoded.coordinates, fine))
            _add(totals["clean_to_clean"], _counts(_selected(decoded, args.threshold), fine))

            # Preserve the same clean-derived coarse site coordinates while
            # removing all information carried by their learned features.
            import MinkowskiEngine as ME
            zero_latent = ME.SparseTensor(
                features=torch.zeros_like(latent.F), coordinates=latent.C,
                tensor_stride=latent.tensor_stride)
            zero_decoded = model.decode(zero_latent, fine_targets=None,
                                        threshold=args.threshold)
            _add(totals["zero_latent_features"],
                 _counts(_selected(zero_decoded, args.threshold), fine))

            surviving, removed, removed_count = remove_coarse_blocks(
                row["cells"], fraction=args.removed_block_fraction,
                seed=3103 + index)
            surviving_coords = _batched(surviving, torch.device("cuda"))
            removed_coords = _batched(removed, torch.device("cuda"))
            masked_latent = model.encode(surviving_coords)
            masked_decoded = model.decode(masked_latent, fine_targets=None,
                                          threshold=args.threshold)
            masked_prediction = _selected(masked_decoded, args.threshold)
            _add(totals["block_masked_to_clean"], _counts(masked_prediction, fine))
            missing_counts = _counts(masked_prediction, removed_coords)
            removed_tp += missing_counts["tp"]
            removed_target += missing_counts["target"]
            fine_sites += len(fine)
            latent_sites += latent.F.shape[0]
            removed_blocks += removed_count
    report = {
        "checkpoint_epoch": int(saved["epoch"]),
        "frames": len(data),
        "latent_channels": int(latent.F.shape[1]),
        "latent_stride_xyz": [int(x) for x in latent.tensor_stride],
        "fine_sites": fine_sites,
        "latent_sites": latent_sites,
        "fine_per_latent_site": fine_sites / latent_sites,
        "candidate_sites_per_latent_site": candidate_counts["predicted"] / latent_sites,
        "unthresholded_candidate_metrics": summarize_counts(candidate_counts),
        "removed_block_fraction": args.removed_block_fraction,
        "removed_blocks": removed_blocks,
        "removed_target_voxels": removed_target,
        "removed_voxels_recovered": removed_tp,
        "removed_voxel_recall": removed_tp / removed_target,
        "metrics": {name: summarize_counts(counts)
                    for name, counts in totals.items()},
        "target_free_decode_identical": True,
    }
    history = args.checkpoint.parent / "history.jsonl"
    if history.is_file():
        for line in history.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if int(record["epoch"]) == report["checkpoint_epoch"]:
                report["saved_epoch_metrics_full_precision"] = {
                    "train_bce": record["train"]["loss"],
                    "val_bce": record["val"]["loss"],
                    "val_tp": record["val"]["tp"],
                    "val_predicted": record["val"]["predicted"],
                    "val_target": record["val"]["target"],
                }
                break
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Report: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
