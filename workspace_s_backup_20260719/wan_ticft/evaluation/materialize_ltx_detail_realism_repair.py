"""Repair generator unrealness by blending only color-matched high-frequency detail.

Full RGB outputs from video generators can be sharp but distribution-shifted:
road texture, building facades, shadows, and small actors look synthetic. This
post-process keeps the low-frequency color/geometry of a trusted base video and
borrows only a small amount of high-frequency detail from the generated video.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from wan_ticft.evaluation.run_sa2va_actor_aware_depth_gate import read_video, write_video


def read_report(path: Path) -> dict[str, Path]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["sample_id"]: Path(row["video"])
            for row in csv.DictReader(handle)
            if row.get("sample_id") and row.get("video") and row.get("status") == "ok"
        }


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "video",
        "status",
        "base_video",
        "detail_video",
        "frames",
        "alpha",
        "sigma",
        "residual_clip",
        "color_match",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def resize_like(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    height, width = reference.shape[:2]
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)


def color_match_frame(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    output = source_lab.copy()
    for channel in range(3):
        src = source_lab[..., channel]
        ref = reference_lab[..., channel]
        src_mean, src_std = float(src.mean()), max(float(src.std()), 1.0)
        ref_mean, ref_std = float(ref.mean()), max(float(ref.std()), 1.0)
        output[..., channel] = (src - src_mean) * (ref_std / src_std) + ref_mean
    output = np.clip(output, 0, 255).astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_LAB2RGB)


def repair_frame(
    base: np.ndarray,
    detail: np.ndarray,
    alpha: float,
    sigma: float,
    residual_clip: float,
    color_match: str,
) -> np.ndarray:
    detail = resize_like(detail, base)
    if color_match == "frame":
        detail = color_match_frame(detail, base)

    base_f = base.astype(np.float32)
    detail_f = detail.astype(np.float32)
    base_low = cv2.GaussianBlur(base_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    detail_low = cv2.GaussianBlur(detail_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    base_high = base_f - base_low
    detail_high = detail_f - detail_low
    residual = np.clip(detail_high - base_high, -residual_clip, residual_clip)
    return np.clip(base_f + alpha * residual, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--detail_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--label_prefix", default="realism")
    parser.add_argument("--alpha", type=float, action="append", required=True)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--residual_clip", type=float, default=18.0)
    parser.add_argument("--color_match", choices=["none", "frame"], default="frame")
    parser.add_argument("--sample_id", action="append", default=[])
    args = parser.parse_args()

    base_report = read_report(args.base_report)
    detail_report = read_report(args.detail_report)
    sample_ids = args.sample_id or sorted(set(base_report) & set(detail_report))
    if not sample_ids:
        raise RuntimeError("no overlapping samples between base and detail reports")

    reports: dict[str, list[dict[str, object]]] = {
        f"{args.label_prefix}_a{int(round(alpha * 100)):02d}": [] for alpha in args.alpha
    }
    for position, sample_id in enumerate(sample_ids, start=1):
        if sample_id not in base_report or sample_id not in detail_report:
            print(f"[{position}/{len(sample_ids)}] {sample_id} | skipped missing report", flush=True)
            continue
        base_frames, fps = read_video(base_report[sample_id])
        detail_frames, _ = read_video(detail_report[sample_id])
        frames = min(len(base_frames), len(detail_frames))
        if frames <= 0:
            print(f"[{position}/{len(sample_ids)}] {sample_id} | skipped empty video", flush=True)
            continue
        if len(base_frames) != len(detail_frames):
            print(
                f"[{position}/{len(sample_ids)}] {sample_id} | length mismatch "
                f"base={len(base_frames)} detail={len(detail_frames)}; using {frames}",
                flush=True,
            )
        else:
            print(f"[{position}/{len(sample_ids)}] {sample_id} | frames={frames}", flush=True)

        base_frames = base_frames[:frames]
        detail_frames = detail_frames[:frames]
        for alpha in args.alpha:
            label = f"{args.label_prefix}_a{int(round(alpha * 100)):02d}"
            output_frames = [
                repair_frame(base, detail, alpha, args.sigma, args.residual_clip, args.color_match)
                for base, detail in zip(base_frames, detail_frames)
            ]
            video_path = args.out_dir / label / "videos" / f"{sample_id}.avi"
            write_video(video_path, output_frames, fps)
            reports[label].append(
                {
                    "sample_id": sample_id,
                    "video": str(video_path),
                    "status": "ok",
                    "base_video": str(base_report[sample_id]),
                    "detail_video": str(detail_report[sample_id]),
                    "frames": frames,
                    "alpha": alpha,
                    "sigma": args.sigma,
                    "residual_clip": args.residual_clip,
                    "color_match": args.color_match,
                    "notes": "",
                }
            )
            write_report(args.out_dir / label / "report.csv", reports[label])
    for label in reports:
        print(f"{label}={args.out_dir / label / 'report.csv'}")


if __name__ == "__main__":
    main()
