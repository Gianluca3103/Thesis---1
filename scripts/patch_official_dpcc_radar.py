"""Apply radar channels and build-only Torch compatibility to pinned D-PCC.

No encoder, decoder geometry, sampling, entropy model, or loss code is changed.
The upstream code is fetched separately because it has no repository license file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


UPSTREAM_REV = "d738f59284e505166d26127dbe21cc71e2fa62c9"
UPSTREAM_URL = "https://github.com/yunhe20/D-PCC.git"
COMPAT_CPP = (
    "ballquery/ballquery_cuda.cpp",
    "featuredistribute/featuredistribute_cuda.cpp",
    "grouping/grouping_cuda.cpp",
    "grouping_int/grouping_int_cuda.cpp",
    "interpolation/interpolation_cuda.cpp",
    "knnquery/knnquery_cuda.cpp",
    "knnquery_heap/knnquery_heap_cuda.cpp",
    "labelstat/labelstat_cuda.cpp",
    "sampling/sampling_cuda.cpp",
)


def adapt(root: Path) -> dict:
    root = Path(root)
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_REV:
        raise ValueError(f"D-PCC revision differs: {actual}")
    edits = {
        "models/layer.py": (
            "            self.out_fdim = 3\n",
            "            self.out_fdim = getattr(args, 'radar_attribute_dim', 3)\n",
        ),
        "models/autoencoder.py": (
            "        return decompressed_xyzs, loss, loss_items, bpp\n",
            "        return decompressed_xyzs, pred_normals, loss, loss_items, bpp\n",
        ),
    }
    result = {"upstream_url": UPSTREAM_URL, "upstream_revision": actual,
              "changes": {}, "build_compatibility": []}
    for relative, (old, new) in edits.items():
        path = root / relative
        source = path.read_text(encoding="utf-8")
        if source.count(old) == 1:
            path.write_text(source.replace(old, new), encoding="utf-8")
        elif source.count(new) != 1:
            raise RuntimeError(f"Unexpected upstream source: {path}")
        result["changes"][relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "description": "radar attribute width" if relative.endswith("layer.py")
                           else "expose decoded attributes for evaluation",
        }
    # PyTorch 2.5 removed the THC header and THCState declaration. These are
    # unused in upstream pointops; remove only those declarations, not kernels.
    for relative in COMPAT_CPP:
        path = root / "models/pointops/src" / relative
        source = path.read_text(encoding="utf-8")
        changed = source.replace("#include <THC/THC.h>\n", "")
        changed = changed.replace("extern THCState *state;\n", "")
        if changed != source:
            path.write_text(changed, encoding="utf-8")
        result["build_compatibility"].append(f"models/pointops/src/{relative}")
    chamfer = root / "models/Chamfer3D/chamfer3D.cu"
    source = chamfer.read_text(encoding="utf-8")
    changed = source.replace(".data<float>()", ".data_ptr<float>()")
    changed = changed.replace(".data<int>()", ".data_ptr<int>()")
    if changed != source:
        chamfer.write_text(changed, encoding="utf-8")
    result["build_compatibility"].append("models/Chamfer3D/chamfer3D.cu")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = adapt(args.root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
