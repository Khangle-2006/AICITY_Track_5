from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    resize_like,
    video_fps,
)


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_targets(cv2, row: dict):
    return [
        image
        for raw in ((row.get("frames") or {}).get("target") or [])
        if (image := read_rgb(cv2, Path(raw))) is not None
    ]


def score(cv2, preds, targets):
    psnrs, ssims, diffs = [], [], []
    for pred, target in zip(preds, targets):
        pred = resize_like(cv2, pred, target)
        psnrs.append(psnr(cv2, np, pred, target))
        ssims.append(ssim_gray(cv2, np, pred, target))
        diffs.append(mean_absdiff(np, pred, target))
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(diffs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether a report is temporally shifted relative to validation targets.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate", default="candidate")
    parser.add_argument("--min_shift", type=int, default=-8)
    parser.add_argument("--max_shift", type=int, default=8)
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    reports = read_report(args.report)
    rows = [row for row in read_manifest(args.manifest) if row["sample_id"] in reports]
    print(f"candidate={args.candidate} rows={len(rows)} report={args.report}")
    print(f"{'shift':>6} {'N':>4} {'PSNR':>9} {'SSIM':>8} {'AbsDiff':>9}")
    print("-" * 44)
    best = None
    for shift in range(args.min_shift, args.max_shift + 1):
        sample_scores = []
        for row in rows:
            video = reports[row["sample_id"]]
            offsets = [max(0, offset + shift) for offset in map_target_offsets(row, video_fps(cv2, video))]
            targets = load_targets(cv2, row)
            frame_map = read_video_rgb_frames(cv2, video, offsets)
            preds = [frame_map[index] for index in offsets if index in frame_map]
            if len(preds) != len(targets) or not targets:
                continue
            sample_scores.append(score(cv2, preds, targets))
        if not sample_scores:
            print(f"{shift:>6} {'0':>4}")
            continue
        arr = np.asarray(sample_scores, dtype=np.float64)
        mean = arr.mean(axis=0)
        print(f"{shift:>6} {len(sample_scores):>4} {mean[0]:>9.4f} {mean[1]:>8.4f} {mean[2]:>9.4f}")
        if best is None or mean[2] < best[1][2]:
            best = (shift, mean, len(sample_scores))
    if best is not None:
        shift, mean, n = best
        print(f"best_by_absdiff shift={shift} N={n} PSNR={mean[0]:.4f} SSIM={mean[1]:.4f} AbsDiff={mean[2]:.4f}")


if __name__ == "__main__":
    main()
