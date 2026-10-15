"""Train independent native Stage-I sparse encoders before Phase 1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage1.online_augmentation import GeometricAugmentationConfig
from models.radar_lidar_stage1.radar_fov import RADAR_FOV_SPEC

from .encoder import BOTTLENECK_CHANNELS
from .previous_encoder import PreviousStage1Sparse, _selected_coarse_sites
from .history import log_record, open_writer, overfitting_warning
from .plots import update_plots
from .pipeline_data import collate_stage1, paired_dataset
from .pretrain_loss import observed_child_distribution, observed_distribution_loss
from .train_teacher import _move


class GeometryPretrainer(nn.Module):
    def __init__(self, modality: str):
        super().__init__()
        self.encoder = PreviousStage1Sparse(modality=modality)
        # Training-only head; neither teacher nor student loads or exports it.
        self.local_distribution = nn.Linear(BOTTLENECK_CHANNELS, 512)

    def forward(self, points, valid, mask_rate, mask_salt):
        sites = self.encoder(points, valid, input_mask_rate=mask_rate, mask_salt=mask_salt)
        target = observed_child_distribution(sites, points, valid)
        logits = self.local_distribution(sites.features)
        held_out = _selected_coarse_sites(sites.coords, mask_rate, mask_salt)
        loss, eligible = observed_distribution_loss(logits, target, held_out)
        return loss, eligible, len(sites.coords)


def _write_progress(path, **values):
    if path is None:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(values), encoding="utf-8")
    os.replace(temporary, path)


def _epoch(model, loader, modality, device, mask_rate, salt, optimizer=None, epoch=0,
           progress_file=None, quiet_progress=False, progress_offset=0,
           progress_total=None):
    training = optimizer is not None
    phase = "train" if training else "val"
    model.train(training)
    loss_sum = 0.
    eligible_sum = active_sum = 0
    total = len(loader)
    display_total = progress_total if progress_total is not None else total
    _write_progress(progress_file, epoch=epoch, phase=phase,
                    step=progress_offset, total=display_total)
    last_progress = time.monotonic()
    for step, raw in enumerate(tqdm(loader, desc=f"{modality} {phase} {epoch:03d}",
                                    unit="batch", dynamic_ncols=True,
                                    disable=quiet_progress, leave=training), 1):
        batch = _move(raw, device)
        name = "radar" if modality == "radar" else "clean_lidar"
        with torch.set_grad_enabled(training):
            loss, eligible, active = model(batch[name], batch[name + "_valid"],
                                           mask_rate, salt)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        loss_sum += float(loss.detach()) * eligible
        eligible_sum += eligible
        active_sum += active
        now = time.monotonic()
        if progress_file is not None and (now - last_progress >= 0.5 or step == total):
            _write_progress(progress_file, epoch=epoch, phase=phase,
                            step=progress_offset + step, total=display_total)
            last_progress = now
    return {"loss_total": loss_sum / eligible_sum if eligible_sum else None,
            "eligible_sites": eligible_sum, "active_sites": active_sum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--modality", choices=("radar", "lidar"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--online-augmentation-config", type=Path,
                        default=Path("configs/stage1_online_geometric_augmentation.json"))
    parser.add_argument("--augmentation-seed", type=int, default=0)
    parser.add_argument("--mask-rate", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--quick-val-frames", type=int, default=64)
    parser.add_argument("--full-val-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.progress_file is not None:
        args.progress_file.parent.mkdir(parents=True, exist_ok=True)
    if args.device != "cuda" or not torch.cuda.is_available():
        parser.error("Full Stage-I sparse encoder training needs CUDA")
    if not 0 < args.mask_rate < 1:
        parser.error("Mask rate must lie strictly between zero and one")
    augmentation = GeometricAugmentationConfig.from_dict(
        json.loads(args.online_augmentation_config.read_text(encoding="utf-8"))
        if args.online_augmentation_config else None)
    train = paired_dataset(args.vod_root, "train", args.radar_variant,
                           augmentation=augmentation, seed=args.augmentation_seed,
                           limit=args.train_limit, data_cache_root=args.data_cache_root,
                           include_clean=args.modality == "lidar" or augmentation.enabled)
    val = paired_dataset(args.vod_root, "val", args.radar_variant,
                         limit=args.val_limit, data_cache_root=args.data_cache_root,
                         include_clean=args.modality == "lidar")
    loaders = {
        "train": DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_stage1, pin_memory=True),
        "quick": DataLoader(Subset(val, list(range(min(args.quick_val_frames, len(val))))),
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate_stage1, pin_memory=True),
        "full": DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers,
                           collate_fn=collate_stage1, pin_memory=True),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "metrics.jsonl").exists() and not args.resume:
        parser.error("Existing pretraining metrics require --resume or a new output directory")
    writer = open_writer(args.output_root, args.tensorboard)
    model = GeometryPretrainer(args.modality).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start, best = 1, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("version") != "previous_stage1_native_sparse_geometry_pretrain_v1"
                or saved.get("modality") != args.modality
                or saved.get("mask_rate") != args.mask_rate
                or saved.get("augmentation") != augmentation.to_dict()
                or (args.modality == "lidar" and saved.get("lidar_fov") != RADAR_FOV_SPEC)):
            raise ValueError("Encoder pretraining resume configuration differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, best = saved["epoch"] + 1, saved["best_val_loss"]
    for epoch in range(start, args.epochs + 1):
        train.set_epoch(epoch)
        full = epoch % args.full_val_every == 0 or epoch == args.epochs
        validation_loader = loaders["full" if full else "quick"]
        progress_total = len(loaders["train"]) + len(validation_loader)
        training = _epoch(model, loaders["train"], args.modality, args.device,
                          args.mask_rate, epoch, optimizer, epoch,
                          args.progress_file, args.quiet_progress,
                          progress_total=progress_total)
        scheduler.step()
        validation = _epoch(model, validation_loader, args.modality,
                            args.device, args.mask_rate, 0, epoch=epoch,
                            progress_file=args.progress_file,
                            quiet_progress=args.quiet_progress,
                            progress_offset=len(loaders["train"]),
                            progress_total=progress_total)
        gap = (validation["loss_total"] - training["loss_total"]
               if validation["loss_total"] is not None and training["loss_total"] is not None else None)
        record = {"epoch": epoch, "train": training, "val": validation,
                  "validation_scope": "full" if full else "quick",
                  "loss_gap": gap,
                  "relative_loss_gap": gap / (abs(training["loss_total"]) + 1e-8) if gap is not None else None}
        with (args.output_root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        log_record(writer, record)
        update_plots(args.output_root, "pretrain")
        warning = overfitting_warning(args.output_root / "metrics.jsonl")
        if warning:
            print(f"Overfitting warning: {warning}", flush=True)
        checkpoint = {"version": "previous_stage1_native_sparse_geometry_pretrain_v1",
                      "modality": args.modality, "mask_rate": args.mask_rate,
                      "augmentation": augmentation.to_dict(),
                      "lidar_fov": RADAR_FOV_SPEC if args.modality == "lidar" else None,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch,
                      "best_val_loss": best, "validation": validation,
                      "encoder_architecture": "stage1_native_sparse_16_32_64_128_growth4_to_64"}
        if full and validation["loss_total"] is not None and validation["loss_total"] < best:
            best = validation["loss_total"]
            checkpoint["best_val_loss"] = best
            torch.save(checkpoint, args.output_root / "best_pretrain.ckpt")
            # Export point encoder, sparse backbone, and 128→64 adapter.
            # The temporary distribution head is not part of inference.
            torch.save(model.encoder.state_dict(), args.output_root / "best_encoder.pth")
        torch.save(checkpoint, args.output_root / "last.ckpt")
        print(f"Previous Stage-I native sparse {args.modality} {epoch:03d}/{args.epochs:03d} "
              f"train {training['loss_total']} val {validation['loss_total']} "
              f"scope {'full' if full else 'quick'}", flush=True)
        _write_progress(args.progress_file, epoch=epoch, phase="complete",
                        step=1, total=1, train_loss=training["loss_total"],
                        val_loss=validation["loss_total"],
                        validation_scope="full" if full else "quick",
                        warning=warning)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
