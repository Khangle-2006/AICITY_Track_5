"""Interpolate an existing test candidate toward its base without rerunning inference."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from baseline.frame_io import numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--candidate_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.ratio <= 1.0:
        raise ValueError("--ratio must be in [0, 1]")
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.test_csv.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["view_type"] == args.view_type]
    for position, row in enumerate(rows, start=1):
        name = row["source_folder"]
        expected = int(row["target_N"])
        base_paths = numeric_pngs(args.base_dir / name)
        candidate_paths = numeric_pngs(args.candidate_dir / name)
        if len(base_paths) != expected or len(candidate_paths) != expected:
            raise RuntimeError(f"{name}: base={len(base_paths)} candidate={len(candidate_paths)} expected={expected}")
        destination = args.out_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        for index, (base_path, candidate_path) in enumerate(zip(base_paths, candidate_paths)):
            base = read_rgb(cv2, base_path)
            candidate = read_rgb(cv2, candidate_path)
            if base is None or candidate is None:
                raise RuntimeError(f"cannot read frame {name}/{index}")
            output = base.astype(np.float32) + args.ratio * (candidate.astype(np.float32) - base.astype(np.float32))
            if not write_rgb(cv2, destination / f"{index}.png", np.clip(output + 0.5, 0, 255).astype(np.uint8)):
                raise RuntimeError(f"cannot write frame {name}/{index}")
        verified = verify_sample_output(destination, expected)
        if not verified.ok:
            raise RuntimeError(f"{name}: {verified.notes}")
        print(f"[{position}/{len(rows)}] {name} | ok", flush=True)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
