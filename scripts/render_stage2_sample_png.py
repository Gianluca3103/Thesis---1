"""Render existing Stage-II sample NPZ files as static PNGs without a browser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import zlib

import numpy as np


MINIMUM = np.array([0.0, -16.0, -2.0], dtype=np.float32)
MAXIMUM = np.array([32.0, 16.0, 2.0], dtype=np.float32)
VOXEL = np.array([0.2, 0.2, 0.25], dtype=np.float32)
SHAPE_ZYX = (16, 160, 160)
PANEL = 360
BAR = 28
BACKGROUND = (12, 22, 32)
COLORS = {
    "clean": (91, 218, 149),
    "faulty": (255, 187, 94),
    "muted": (90, 108, 126),
    "radar": (91, 173, 255),
    "predicted": (255, 125, 91),
    "eligible": (74, 52, 112),
    "target": (91, 218, 149),
    "false_positive": (247, 82, 112),
    "true_positive": (255, 255, 255),
}

# A tiny built-in font keeps this renderer independent of Pillow, Matplotlib,
# Firefox, and any image or display libraries on the professor machine.
FONT = {
    " ": ("00000",) * 7,
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01111", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "11110"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00100", "00100"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    "/": ("00001", "00001", "00010", "00100", "01000", "10000", "10000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "%": ("11001", "11010", "00100", "00100", "01011", "10011", "00000"),
}


def _write_png(path: Path, pixels: np.ndarray) -> None:
    height, width, channels = pixels.shape
    if pixels.dtype != np.uint8 or channels != 3:
        raise ValueError("Expected RGB uint8 image")

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\0" + row.tobytes() for row in pixels)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height,
                                                    8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, level=6))
                     + chunk(b"IEND", b""))


def _text(canvas: np.ndarray, value: str, x: int, y: int,
          color: tuple[int, int, int] = (232, 239, 246), scale: int = 2) -> None:
    for character in value.upper():
        glyph = FONT.get(character, FONT[" "])
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    canvas[y + row * scale:y + (row + 1) * scale,
                           x + column * scale:x + (column + 1) * scale] = color
        x += 6 * scale


def _plot(canvas: np.ndarray, xyz: np.ndarray, *, side: bool,
          color: tuple[int, int, int], radius: int = 0) -> None:
    if not len(xyz):
        return
    axes = (0, 2) if side else (0, 1)
    minimum = MINIMUM[list(axes)]
    maximum = MAXIMUM[list(axes)]
    xy = xyz[:, list(axes)]
    valid = np.isfinite(xy).all(1) & (xy >= minimum).all(1) & (xy < maximum).all(1)
    xy = xy[valid]
    if not len(xy):
        return
    height, width, _ = canvas.shape
    px = np.clip(((xy[:, 0] - minimum[0]) / (maximum[0] - minimum[0])
                  * (width - 1)).astype(np.int32), 0, width - 1)
    py = np.clip(((maximum[1] - xy[:, 1]) / (maximum[1] - minimum[1])
                  * (height - 1)).astype(np.int32), 0, height - 1)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xx, yy = px + dx, py + dy
            keep = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            canvas[yy[keep], xx[keep]] = color


def _keys(xyz: np.ndarray) -> np.ndarray:
    inside = ((xyz >= MINIMUM) & (xyz < MAXIMUM)).all(1)
    ijk = np.floor((xyz[inside] - MINIMUM) / VOXEL).astype(np.int64)
    return (ijk[:, 2] * SHAPE_ZYX[1] * SHAPE_ZYX[2]
            + ijk[:, 1] * SHAPE_ZYX[2] + ijk[:, 0])


def _centers(keys: np.ndarray) -> np.ndarray:
    z = keys // (SHAPE_ZYX[1] * SHAPE_ZYX[2])
    y = (keys // SHAPE_ZYX[2]) % SHAPE_ZYX[1]
    x = keys % SHAPE_ZYX[2]
    ijk = np.stack((x, y, z), axis=1)
    return (ijk + 0.5) * VOXEL + MINIMUM


def _panel(label: str, *, side: bool, layers: list[tuple[np.ndarray, str, int]]) -> np.ndarray:
    panel = np.full((PANEL + BAR, PANEL, 3), BACKGROUND, dtype=np.uint8)
    panel[:BAR] = (27, 42, 57)
    _text(panel, label, 8, 7)
    plot = panel[BAR:]
    for xyz, name, radius in layers:
        _plot(plot, xyz, side=side, color=COLORS[name], radius=radius)
    return panel


def render_sample(archive: Path, output: Path) -> tuple[np.ndarray, dict]:
    with np.load(archive, allow_pickle=False) as item:
        clean = item["clean_xyz"].astype(np.float32)
        faulty = item["faulty_xyz"].astype(np.float32)
        predicted = item["xyz"].astype(np.float32)
        eligible = item["allowed_xyz"].astype(np.float32)
        radar = (item["radar_xyz"].astype(np.float32)
                 if "radar_xyz" in item else np.empty((0, 3), dtype=np.float32))
        fault = str(item["fault_type"])
    if any(array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all()
           for array in (clean, faulty, predicted, eligible, radar)):
        raise ValueError(f"Invalid XYZ arrays: {archive}")
    target_keys = np.intersect1d(np.unique(_keys(clean)), np.unique(_keys(eligible)))
    predicted_keys = np.unique(_keys(predicted))
    tp_keys = np.intersect1d(predicted_keys, target_keys)
    fp_keys = np.setdiff1d(predicted_keys, target_keys)
    iou = len(tp_keys) / max(len(target_keys) + len(predicted_keys) - len(tp_keys), 1)
    target_xyz, tp_xyz, fp_xyz = map(_centers, (target_keys, tp_keys, fp_keys))
    labels = ("CLEAN", "FAULTY", "FAULTY + RADAR + RECON", "ELIGIBLE / TP / FP")
    layers = (
        [(clean, "clean", 0)],
        [(faulty, "faulty", 0)],
        [(faulty, "muted", 0), (radar, "radar", 1),
         (predicted, "predicted", 1)],
        [(eligible, "eligible", 0), (target_xyz, "target", 1),
         (fp_xyz, "false_positive", 1), (tp_xyz, "true_positive", 1)],
    )
    canvas = np.full((2 * (PANEL + BAR), 4 * PANEL, 3), BACKGROUND,
                     dtype=np.uint8)
    for column, (label, items) in enumerate(zip(labels, layers)):
        canvas[:PANEL + BAR, column * PANEL:(column + 1) * PANEL] = _panel(
            "TOP " + label, side=False, layers=items)
        canvas[PANEL + BAR:, column * PANEL:(column + 1) * PANEL] = _panel(
            "SIDE " + label, side=True, layers=items)
    _write_png(output, canvas)
    preview = canvas[:PANEL + BAR, 3 * PANEL:4 * PANEL].copy()
    preview[:BAR] = (27, 42, 57)
    frame_id = archive.stem.split("_", 1)[0]
    _text(preview, f"{frame_id} {fault} {iou:.1%}", 8, 7)
    return preview, {
        "sample": archive.name, "fault": fault,
        "predicted": len(predicted_keys), "eligible_target": len(target_keys),
        "exact_tp": len(tp_keys), "exact_iou": iou,
        "png": output.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    archives = sorted(args.sample_root.glob("*.npz"))
    if not archives:
        parser.error(f"No Stage-II sample NPZ files under {args.sample_root}")
    output_root = args.output_root or args.sample_root / "static_png"
    output_root.mkdir(parents=True, exist_ok=True)
    previews, rows = [], []
    for archive in archives:
        preview, row = render_sample(
            archive, output_root / f"{archive.stem}_top_side.png")
        previews.append(preview)
        rows.append(row)
        print(f"{row['png']}: fault={row['fault']}, "
              f"exact eligible IoU={row['exact_iou']:.2%}", flush=True)
    sheet = np.full((2 * (PANEL + BAR), 4 * PANEL, 3), BACKGROUND, dtype=np.uint8)
    for index, preview in enumerate(previews[:8]):
        top, left = divmod(index, 4)
        sheet[top * (PANEL + BAR):(top + 1) * (PANEL + BAR),
              left * PANEL:(left + 1) * PANEL] = preview
    _write_png(output_root / "contact_sheet.png", sheet)
    (output_root / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Open in VS Code: {output_root / 'contact_sheet.png'}", flush=True)


if __name__ == "__main__":
    main()
