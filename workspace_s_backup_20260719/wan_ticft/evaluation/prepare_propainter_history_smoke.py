"""Prepare Sa2VA-masked BDD histories for a ProPainter background-plate smoke test."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--sample_id", action="append", required=True)
    parser.add_argument("--pre_dilate", type=int, default=5)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.sample_id)
    rows = {row["sample_id"]: row for row in read_jsonl(args.manifest) if row.get("sample_id") in wanted}
    if len(rows) != len(wanted):
        raise RuntimeError(f"requested={len(wanted)} found={len(rows)}")
    kernel = np.ones((args.pre_dilate, args.pre_dilate), np.uint8)
    for sample_id in args.sample_id:
        paths = (rows[sample_id].get("frames") or {}).get("condition") or []
        mask_path = args.mask_root / sample_id / "masks.npy"
        if len(paths) != 4 or not mask_path.is_file():
            raise RuntimeError(f"{sample_id}: frames={len(paths)} masks={mask_path}")
        masks = np.load(mask_path).astype(bool)
        if len(masks) != len(paths):
            raise RuntimeError(f"{sample_id}: masks={len(masks)} frames={len(paths)}")
        frame_dir = args.out_dir / "inputs" / sample_id
        mask_dir = args.out_dir / "masks" / sample_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for index, (path, mask) in enumerate(zip(paths, masks)):
            frame = np.asarray(Image.open(path).convert("RGB"))
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) * 255
            Image.fromarray(frame).save(frame_dir / f"{index:04d}.png")
            Image.fromarray(mask).save(mask_dir / f"{index:04d}.png")
        print(f"{sample_id} | frames=4", flush=True)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
