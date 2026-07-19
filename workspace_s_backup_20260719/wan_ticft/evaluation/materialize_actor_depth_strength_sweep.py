"""Materialize stronger actor-protected depth variants from cached full-video Sa2VA masks."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from wan_ticft.evaluation.run_sa2va_actor_aware_depth_gate import (
    composite,
    read_report,
    read_video,
    soften,
    write_video,
)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--depth_report", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--strength", action="append", required=True, help="name:scale_from_a75")
    parser.add_argument("--dilation", type=int, default=15)
    parser.add_argument("--blur_sigma", type=float, default=3.0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    variants = {}
    for value in args.strength:
        name, raw_scale = value.split(":", 1)
        variants[name] = float(raw_scale)
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifest)
    bases = read_report(args.base_report)
    depths = read_report(args.depth_report)
    reports = {name: [] for name in variants}

    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        base_frames, fps = read_video(bases[sample_id])
        depth_frames, _ = read_video(depths[sample_id])
        mask_path = args.mask_dir / f"{sample_id}_masks.npz"
        if not mask_path.is_file():
            raise RuntimeError(f"missing masks: {mask_path}")
        archive = np.load(mask_path)
        masks = [archive[key].astype(bool) for key in sorted(archive.files)]
        if len(base_frames) != len(depth_frames) or len(base_frames) != len(masks):
            raise RuntimeError(
                f"{sample_id}: base={len(base_frames)} depth={len(depth_frames)} masks={len(masks)}"
            )
        height, width = base_frames[0].shape[:2]
        depth_frames = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in depth_frames]
        soft_masks = [soften(mask, args.dilation, args.blur_sigma) for mask in masks]
        for name, scale in variants.items():
            outputs = [
                composite(base, depth, mask, scale)
                for base, depth, mask in zip(base_frames, depth_frames, soft_masks)
            ]
            path = args.out_dir / name / "videos" / f"{sample_id}.avi"
            write_video(path, outputs, fps)
            reports[name].append(
                {"sample_id": sample_id, "video": str(path), "status": "ok", "future_frames": len(outputs)}
            )
        print(f"[{position}/{len(rows)}] {sample_id} | ok", flush=True)

    for name, report_rows in reports.items():
        path = args.out_dir / name / "report.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
            writer.writeheader()
            writer.writerows(report_rows)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
