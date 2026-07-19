"""Blend only the temporally varying part of a color-matched detailer residual."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from wan_ticft.evaluation.materialize_detailer_residual_blends import global_color_match
from wan_ticft.evaluation.run_sa2va_actor_aware_depth_gate import read_video, write_video


def report_video(path: Path, sample_id: str) -> Path:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("sample_id") == sample_id and row.get("status") == "ok":
                return Path(row["video"])
    raise RuntimeError(f"missing {sample_id} in {path}")


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--detailer_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--sample_id", action="append", required=True)
    parser.add_argument("--alpha", type=float, action="append", required=True)
    args = parser.parse_args()

    reports = {f"tmblend_a{int(round(alpha * 100)):02d}": [] for alpha in args.alpha}
    for sample_id in args.sample_id:
        base_frames, fps = read_video(report_video(args.base_report, sample_id))
        detail_frames, _ = read_video(report_video(args.detailer_report, sample_id))
        if len(base_frames) != len(detail_frames):
            raise RuntimeError(f"{sample_id}: base={len(base_frames)} detailer={len(detail_frames)} frames")

        height, width = base_frames[0].shape[:2]
        detail_frames = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC) for frame in detail_frames]
        matched = np.stack(global_color_match(detail_frames, base_frames)).astype(np.float32)
        base = np.stack(base_frames).astype(np.float32)

        residual = matched - base
        static_residual = np.median(residual, axis=0, keepdims=True)
        temporal_residual = residual - static_residual

        for alpha in args.alpha:
            tag = f"tmblend_a{int(round(alpha * 100)):02d}"
            output = np.clip(base + alpha * temporal_residual, 0, 255).astype(np.uint8)
            path = args.out_dir / tag / f"{sample_id}.avi"
            write_video(path, list(output), fps)
            reports[tag].append({"sample_id": sample_id, "video": str(path), "status": "ok"})
            write_report(args.out_dir / tag / "report.csv", reports[tag])
            print(f"{tag}={path}")


if __name__ == "__main__":
    main()
