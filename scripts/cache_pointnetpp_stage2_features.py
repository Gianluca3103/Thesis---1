"""Cache frozen PointNet++ radar anchor features for conditional Stage II."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from scripts.pointnetpp_stage2_radar_data import Stage2RadarFrames, Stage2RadarScale
from scripts.radar_encoder_ablation import RadarAutoencoder, farthest_points, gather
from scripts.train_pointnetpp_stage2_radar import FixedRadarInput, VERSION as AE_VERSION


VERSION = "pointnetpp_radar_anchor_features_stage2_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class RadarFeatureCache:
    def __init__(self, root: Path, split: str, checkpoint_hash: str | None = None):
        self.root = Path(root) / split
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.frame_ids = set(self.manifest.get("frame_ids", []))
        self.feature_channels = self.manifest.get("feature_channels")
        if (self.manifest.get("version") != VERSION or
                self.feature_channels not in (768, 1024) or
                self.manifest.get("anchors") != 512 or
                (checkpoint_hash is not None and
                 self.manifest.get("radar_checkpoint_sha256") != checkpoint_hash)):
            raise ValueError(f"Incompatible radar feature cache: {self.manifest_path}")

    def get(self, frame_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        if frame_id not in self.frame_ids:
            raise ValueError(f"Radar feature frame is outside {self.manifest_path}: {frame_id}")
        path = self.root / f"{frame_id}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        xyz, features = saved.get("xyz"), saved.get("features")
        if (saved.get("version") != VERSION or saved.get("frame_id") != frame_id or
                not isinstance(xyz, torch.Tensor) or xyz.ndim != 2 or xyz.shape[1] != 3 or
                not isinstance(features, torch.Tensor) or
                features.shape != (len(xyz), self.feature_channels) or
                len(xyz) > 512 or not torch.isfinite(xyz).all() or
                not torch.isfinite(features).all()):
            raise ValueError(f"Invalid radar feature frame: {path}")
        return xyz.float(), features.float()


def cache_split(radar_root: Path, checkpoint: Path, output_root: Path,
                split: str, *, limit: int | None = None, device: str = "cuda") -> dict:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = saved["identity"]
    if (identity["version"] != AE_VERSION or
            identity.get("variant", "pointnetpp") != "pointnetpp" or
            identity["width"] not in (768, 1024) or identity["anchors"] != 512 or
            identity["input_points"] != 2048 or identity["children"] != 4 or
            identity.get("dropout_p", 0.0) != 0.0):
        raise ValueError("Expected trained width-768/1024, no-dropout PointNet++ radar AE")
    frames = Stage2RadarFrames(radar_root, split, limit)
    if split == "train" and frames.sha256() != identity["train_manifest"]:
        raise ValueError("Training radar frames differ from checkpoint")
    if split == "val" and frames.sha256() != identity["val_manifest"]:
        raise ValueError("Validation radar frames differ from checkpoint")
    scale = Stage2RadarScale.from_json(identity["scale"])
    model = RadarAutoencoder(
        "pointnetpp", scale.minimum, scale.maximum, width=identity["width"],
        anchors=512, children=4, columns=7).to(device)
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    checkpoint_hash = sha256(checkpoint)
    manifest = {"version": VERSION, "split": split,
                "frame_ids": frames.ids, "radar_manifest_sha256": frames.sha256(),
                "radar_checkpoint_sha256": checkpoint_hash,
                "radar_checkpoint_epoch": saved["epoch"],
                "input_points": 2048, "anchors": 512,
                "feature_channels": identity["width"],
                "feature_dtype": "float16",
                "empty_frame_policy": "empty anchors and features"}
    target = Path(output_root) / split
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Existing feature cache belongs to another run: {manifest_path}")
    else:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    missing = sum(not (target / f"{frame_id}.pt").is_file()
                  for frame_id in frames.ids)
    estimated_bytes = missing * (512 * identity["width"] * 2 + 512 * 3 * 4 + 65536)
    free_bytes = shutil.disk_usage(target).free
    if estimated_bytes + 2 * 1024**3 > free_bytes:
        raise OSError(f"Radar feature cache needs about {estimated_bytes / 1024**3:.1f} "
                      f"GiB plus 2 GiB reserve; only {free_bytes / 1024**3:.1f} GiB free")
    print(f"{split}: {missing} uncached frames, estimated "
          f"{estimated_bytes / 1024**3:.1f} GiB, "
          f"{free_bytes / 1024**3:.1f} GiB free", flush=True)
    source = FixedRadarInput(frames, scale, 2048, training=False)
    written = reused = 0
    for index, frame_id in enumerate(tqdm(frames.ids, desc=f"PointNet++ features {split}",
                                         unit="frame")):
        path = target / f"{frame_id}.pt"
        if path.is_file():
            reused += 1
            continue
        row = source[index]
        if row["usable"]:
            normalized = row["input"].unsqueeze(0).to(device)
            with torch.inference_mode():
                xyz = scale.decode(normalized)[..., :3].contiguous()
                anchors = farthest_points(xyz, 512)
                features = model.encoder(xyz, normalized[..., 3:], anchors)
                centers = gather(xyz, anchors)
            payload = {"version": VERSION, "frame_id": frame_id,
                       "xyz": centers[0].cpu(), "features": features[0].half().cpu()}
        else:
            payload = {"version": VERSION, "frame_id": frame_id,
                       "xyz": torch.empty((0, 3)),
                       "features": torch.empty((0, identity["width"]), dtype=torch.float16)}
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        written += 1
    return {"split": split, "frames": len(frames), "written": written,
            "reused": reused, "checkpoint_epoch": saved["epoch"],
            "checkpoint_sha256": checkpoint_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radar-cache-root", type=Path, required=True)
    parser.add_argument("--radar-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(cache_split(args.radar_cache_root, args.radar_checkpoint,
                                 args.output_root, args.split,
                                 limit=args.limit, device=args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()
