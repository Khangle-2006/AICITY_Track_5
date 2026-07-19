"""Create original/mask/inpaint comparison sheets for the ProPainter history smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def panel(rgb: np.ndarray, label: str) -> np.ndarray:
    tile = cv2.resize(rgb, (384, 216), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (384, 27), (0, 0, 0), -1)
    cv2.putText(tile, label, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample_id", action="append", required=True)
    args = parser.parse_args()

    sheet_dir = args.root / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in args.sample_id:
        originals, overlays, outputs = [], [], []
        for index in range(4):
            original = np.asarray(Image.open(args.root / "inputs" / sample_id / f"{index:04d}.png").convert("RGB"))
            mask = np.asarray(Image.open(args.root / "masks" / sample_id / f"{index:04d}.png").convert("L")) > 0
            output = np.asarray(
                Image.open(args.root / "inpaint" / sample_id / "frames" / f"{index:04d}.png").convert("RGB")
            )
            overlay = original.copy()
            overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
            originals.append(panel(original, f"history {index}"))
            overlays.append(panel(overlay, f"remove mask {index}"))
            outputs.append(panel(output, f"ProPainter {index}"))
        sheet = np.concatenate(
            [np.concatenate(originals, axis=1), np.concatenate(overlays, axis=1), np.concatenate(outputs, axis=1)],
            axis=0,
        )
        Image.fromarray(sheet).save(sheet_dir / f"{sample_id}.jpg", quality=94)
        print(f"ok {sheet_dir / f'{sample_id}.jpg'}")


if __name__ == "__main__":
    main()
