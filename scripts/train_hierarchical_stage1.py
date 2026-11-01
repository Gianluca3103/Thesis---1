"""Train the fault-aware, coarse-to-fine Stage-I coordinate generator.

This is a separate experiment. It scores missing clean cells within Stage II's
ROI and never supplies clean coordinates to inference. The optional oracle
parent retention is training-only deep supervision, not a validation shortcut.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.hierarchical_gate import (
    HierarchicalGateConfig, HierarchicalStage1Gate, checkpoint_payload, fine_ranking,
    missing_target, stage_loss, target_mask)
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.fault_data import (
    SevereFogCache, retain_original_fog_returns, severe_fov_filter)


def _single(rows: list[dict]) -> dict:
    if len(rows) != 1:
        raise ValueError("Hierarchical Stage I currently uses one frame per batch")
    return rows[0]


def faulted(clean: torch.Tensor, name: str, *, frame_id: str,
            fog_cache: SevereFogCache | None = None) -> torch.Tensor:
    if name == "total_loss":
        return clean.new_empty((0, 3))
    if name == "fov_filter":
        return torch.from_numpy(severe_fov_filter(clean[:, :3].numpy()).copy())
    if name == "fog_sim" and fog_cache is not None:
        return torch.from_numpy(retain_original_fog_returns(
            clean.numpy(), fog_cache.load(frame_id).numpy()).copy())
    raise ValueError(f"Unsupported or uncached fault: {name}")


def _train_fault(frame_id: str, epoch: int, modes: tuple[str, ...], seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{frame_id}".encode()).digest()
    start = int.from_bytes(digest[:8], "little") % len(modes)
    return modes[(start + epoch - 1) % len(modes)]


def _validation_indices(frames: list, limit: int, seed: int,
                        strategy: str) -> list[int]:
    rng = random.Random(seed)
    count = min(limit, len(frames))
    if strategy == "random":
        return rng.sample(range(len(frames)), count)
    if strategy != "blocks":
        raise ValueError("Unknown validation selection strategy")
    try:
        ids = [int(frame.frame_id) for frame in frames]
    except ValueError as error:
        raise ValueError("Block validation requires numeric frame IDs") from error
    groups: list[list[int]] = []
    for index, frame_id in enumerate(ids):
        if not groups or frame_id != ids[index - 1] + 1:
            groups.append([])
        groups[-1].append(index)
    allocation = [0] * len(groups)
    while sum(allocation) < count:
        changed = False
        for group_index, group in enumerate(groups):
            if allocation[group_index] < len(group) and sum(allocation) < count:
                allocation[group_index] += 1
                changed = True
        if not changed:
            break
    selected = []
    for group, size in zip(groups, allocation):
        if size:
            start = rng.randrange(len(group) - size + 1)
            selected.extend(group[start:start + size])
    return selected


def _counts(output: dict, truth: torch.Tensor, budget: int) -> dict[str, int]:
    scores, _ = fine_ranking(output)
    selected = torch.topk(scores, min(budget, len(scores))).indices
    predicted = output["fine_coords"][selected]
    tp = int(target_mask(predicted, truth, 1).sum())
    return {"tp": tp, "predicted": len(predicted), "target": len(truth)}


def _rates(counts: dict[str, int]) -> dict[str, float | int]:
    tp, predicted, target = (counts[key] for key in ("tp", "predicted", "target"))
    return {**counts,
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / target if target else 1.0}


def evaluate(model: HierarchicalStage1Gate, loader, device: str,
             modes: tuple[str, ...], budgets: tuple[int, ...],
             fog_cache: SevereFogCache | None = None) -> dict:
    model.eval()
    totals = {name: {budget: {"tp": 0, "predicted": 0, "target": 0}
                     for budget in budgets} for name in modes}
    pool = {name: {"tp": 0, "predicted": 0, "target": 0} for name in modes}
    frame_counts = {name: 0 for name in modes}
    with torch.no_grad():
        for row in tqdm(loader, desc="Hierarchical Stage-I validation", unit="frame"):
            radar = row["radar"].to(device)
            clean = row["clean_lidar"]
            for name in modes:
                faulty = faulted(clean, name, frame_id=row["frame_id"], fog_cache=fog_cache)
                faulty = faulty.to(device)
                truth = missing_target(clean[:, :3].to(device), faulty)
                output = model(radar, faulty)
                entire = output["fine_coords"]
                pool[name]["tp"] += int(target_mask(entire, truth, 1).sum())
                pool[name]["predicted"] += len(entire)
                pool[name]["target"] += len(truth)
                frame_counts[name] += 1
                for budget in budgets:
                    counts = _counts(output, truth, budget)
                    for key in counts:
                        totals[name][budget][key] += counts[key]
    return {name: {"frames": frame_counts[name],
                   "pool_oracle": _rates(pool[name]),
                   "budgets": {str(budget): _rates(totals[name][budget])
                               for budget in budgets}}
            for name in modes}


def _save(path: Path, model, optimizer, scheduler, *, epoch: int,
          radar_variant: str, modes: tuple[str, ...], seed: int,
          validation: dict, selection_score: float,
          selection_frame_ids: tuple[str, ...],
          inherited_selection_frame_ids: tuple[str, ...] = (),
          initialized_from: dict | None = None) -> None:
    payload = checkpoint_payload(
        model, epoch=epoch, radar_variant=radar_variant, fault_modes=modes,
        seed=seed, validation=validation, selection_score=selection_score,
        selection_frame_ids=selection_frame_ids,
        inherited_selection_frame_ids=inherited_selection_frame_ids,
        initialized_from=initialized_from,
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict())
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/hierarchical_stage1_gate_v1.json"))
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--fog-cache-root", type=Path)
    parser.add_argument("--fault-modes", nargs="+", choices=(
        "total_loss", "fov_filter", "fog_sim"), default=("total_loss", "fov_filter"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int, default=128)
    parser.add_argument("--val-selection", choices=("blocks", "random"),
                        default="blocks")
    parser.add_argument("--budgets", type=int, nargs="+", default=(5000, 10000, 20000, 40000))
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path,
                        help="Initialize weights from a compatible gate checkpoint")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("Use --resume or --init-from, not both")
    if (args.epochs < 1 or args.learning_rate <= 0 or args.num_workers < 0
            or args.val_limit < 1 or not args.budgets or min(args.budgets) < 1
            or len(set(args.budgets)) != len(args.budgets)
            or len(set(args.fault_modes)) != len(args.fault_modes)):
        parser.error("Invalid training, fault, or site-budget settings")
    if "fog_sim" in args.fault_modes and args.fog_cache_root is None:
        parser.error("fog_sim requires --fog-cache-root")
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    config = HierarchicalGateConfig(**raw)
    torch.manual_seed(args.seed)
    model = HierarchicalStage1Gate(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    modes = tuple(args.fault_modes)
    train = paired_dataset(args.vod_root, "train", args.radar_variant,
                           data_cache_root=args.data_cache_root)
    if args.train_limit is not None:
        if args.train_limit < 1:
            parser.error("--train-limit must be positive")
        sampled = random.Random(args.seed).sample(
            range(len(train)), min(args.train_limit, len(train)))
        train = Subset(train, sampled)
    val = paired_dataset(args.vod_root, "val", args.radar_variant,
                         data_cache_root=args.data_cache_root)
    selected = _validation_indices(val.frames, args.val_limit, args.seed,
                                   args.val_selection)
    selection_frame_ids = tuple(val.frames[index].frame_id for index in selected)
    val = Subset(val, selected)
    train_loader = DataLoader(train, batch_size=1, shuffle=True,
                              num_workers=args.num_workers, collate_fn=_single)
    val_loader = DataLoader(val, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=_single)
    caches = {}
    if args.fog_cache_root is not None:
        for split in ("train", "val"):
            caches[split] = SevereFogCache(args.fog_cache_root, split, None,
                                            seed=args.seed, check_complete=False)
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = args.output_root / "metrics.jsonl"
    start = 1
    initialized_from = None
    inherited_selection_frame_ids: tuple[str, ...] = ()
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("config") != asdict(config)
                or saved.get("radar_variant") != args.radar_variant
                or tuple(saved.get("fault_modes", ())) != modes
                or saved.get("seed") != args.seed
                or (saved.get("selection_frame_ids") is not None
                    and tuple(saved["selection_frame_ids"]) != selection_frame_ids)):
            raise ValueError("Resume checkpoint has incompatible geometry or fault modes")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = saved["epoch"] + 1
        initialized_from = saved.get("initialized_from")
        inherited_selection_frame_ids = tuple(saved.get(
            "inherited_selection_frame_ids", ()))
    elif args.init_from:
        saved = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if (saved.get("version") != "hierarchical_stage1_gate_v1"
                or saved.get("config", {}).get("channels") != config.channels
                or saved.get("radar_variant") != args.radar_variant):
            raise ValueError("Initialization checkpoint has incompatible model or radar")
        model.load_state_dict(saved["model"], strict=True)
        digest = hashlib.sha256()
        with args.init_from.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        initialized_from = {"path": str(args.init_from.resolve()),
                            "sha256": digest.hexdigest(), "epoch": saved["epoch"]}
        prior_selection = set(saved.get("selection_frame_ids", ()))
        prior_selection.update(saved.get("inherited_selection_frame_ids", ()))
        inherited_selection_frame_ids = tuple(sorted(prior_selection))
    elif history.exists():
        parser.error("Existing metrics require --resume or a new output directory")
    best = -1.0
    best_path = args.output_root / "best.ckpt"
    if best_path.exists():
        previous = torch.load(best_path, map_location="cpu", weights_only=False)
        best = float(previous.get("selection_score", -1.0))
    for epoch in range(start, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        frames = 0
        for row in tqdm(train_loader, desc=f"Hierarchical Stage-I train {epoch:03d}",
                        unit="frame"):
            name = _train_fault(row["frame_id"], epoch, modes, args.seed)
            clean = row["clean_lidar"]
            faulty = faulted(clean, name, frame_id=row["frame_id"],
                             fog_cache=caches.get("train")).to(args.device)
            truth = missing_target(clean[:, :3].to(args.device), faulty)
            radar = row["radar"].to(args.device)
            output = model(radar, faulty, target_fine=truth, teacher_forcing=True)
            loss, _ = stage_loss(output, truth, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            frames += 1
        validation = evaluate(model, val_loader, args.device, modes,
                              tuple(sorted(args.budgets)), caches.get("val"))
        scheduler.step()
        # Select among precision-feasible budgets first. The graded fallback
        # only distinguishes checkpoints before any budget becomes feasible.
        score = 0.0
        for name in modes:
            rows = tuple(validation[name]["budgets"].values())
            feasible = [row["recall"] for row in rows if row["precision"] >= 0.20]
            mode_score = (max(feasible) if feasible else max(
                (row["recall"] * row["precision"] / 0.20 for row in rows),
                default=0.0))
            score += mode_score / len(modes)
        record = {"epoch": epoch, "train_loss": loss_sum / max(frames, 1),
                  "train_frames": frames, "validation": validation,
                  "selection_score": score}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        _save(args.output_root / "last.ckpt", model, optimizer, scheduler,
              epoch=epoch, radar_variant=args.radar_variant, modes=modes,
              seed=args.seed, validation=validation, selection_score=score,
              selection_frame_ids=selection_frame_ids,
              inherited_selection_frame_ids=inherited_selection_frame_ids,
              initialized_from=initialized_from)
        if score > best:
            best = score
            _save(best_path, model, optimizer, scheduler,
                  epoch=epoch, radar_variant=args.radar_variant, modes=modes,
                  seed=args.seed, validation=validation, selection_score=score,
                  selection_frame_ids=selection_frame_ids,
                  inherited_selection_frame_ids=inherited_selection_frame_ids,
                  initialized_from=initialized_from)
        print(f"Epoch {epoch}: loss={record['train_loss']:.4f} score={score:.4f} "
              f"validation={json.dumps(validation)}", flush=True)


if __name__ == "__main__":
    main()
