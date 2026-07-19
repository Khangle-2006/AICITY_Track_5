"""Evaluate low-frequency region correction while preserving LTX detail."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.evaluation.evaluate_bdd_ego_motion import region_predictions
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    video_fps,
)
from wan_ticft.evaluation.evaluate_ltx_region_composite_ceiling import (
    load_images,
    read_jsonl,
    resize_like,
    target_offsets,
)


VARIANTS = (
    ("lowfreq_s8_a25", 8.0, 0.25),
    ("lowfreq_s16_a25", 16.0, 0.25),
    ("lowfreq_s16_a50", 16.0, 0.50),
)


def low_frequency_correct(cv2, np, ltx, region, sigma: float, alpha: float):
    ltx_float = ltx.astype(np.float32)
    region_float = region.astype(np.float32)
    ltx_low = cv2.GaussianBlur(ltx_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    region_low = cv2.GaussianBlur(region_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(ltx_float + alpha * (region_low - ltx_low), 0, 255).astype(np.uint8)


def score_frames(cv2, np, predictions, targets):
    values = [
        (
            psnr(cv2, np, prediction, target),
            ssim_gray(cv2, np, prediction, target),
            mean_absdiff(np, prediction, target),
        )
        for prediction, target in zip(predictions, targets)
    ]
    return tuple(float(np.mean([value[index] for value in values])) for index in range(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--estimate_width", type=int, default=640)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    reports = read_report(args.ltx_report)
    output = []
    for position, row in enumerate(read_jsonl(args.manifest), start=1):
        video = reports.get(row["sample_id"])
        paths = row.get("frames") or {}
        conditions = load_images(cv2, paths.get("condition") or [])
        targets = load_images(cv2, paths.get("target") or [])
        if video is None or len(conditions) < 2 or not targets:
            print(f"[{position}] {row['sample_id']} | skipped", flush=True)
            continue

        indices = (row.get("frame_indices") or {}).get("condition") or []
        history_gap = max(1, int(indices[-1]) - int(indices[-2])) if len(indices) >= 2 else 1
        offsets = target_offsets(row, len(targets))
        region = region_predictions(
            cv2, np, conditions[-2], conditions[-1], offsets, history_gap, args.estimate_width
        )
        mapped = map_target_offsets(row, video_fps(cv2, video))
        frame_map = read_video_rgb_frames(cv2, video, mapped)
        ltx = [frame_map[index] for index in mapped if index in frame_map]
        if len(ltx) != len(targets):
            print(f"[{position}] {row['sample_id']} | incomplete LTX", flush=True)
            continue

        region = [resize_like(cv2, frame, target) for frame, target in zip(region, targets)]
        ltx = [resize_like(cv2, frame, target) for frame, target in zip(ltx, targets)]
        candidates = {"region": region, "ltx16": ltx}
        for name, sigma, alpha in VARIANTS:
            candidates[name] = [
                low_frequency_correct(cv2, np, ltx_frame, region_frame, sigma, alpha)
                for ltx_frame, region_frame in zip(ltx, region)
            ]

        for candidate, frames in candidates.items():
            p, s, d = score_frames(cv2, np, frames, targets)
            output.append(
                {
                    "sample_id": row["sample_id"],
                    "candidate": candidate,
                    "psnr": f"{p:.6f}",
                    "ssim": f"{s:.6f}",
                    "absdiff": f"{d:.6f}",
                }
            )
        print(f"[{position}] {row['sample_id']} | ok", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)

    print(f"{'Candidate':<20} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 60)
    summary = {}
    for candidate in ("region", "ltx16", *(name for name, _, _ in VARIANTS)):
        rows = [row for row in output if row["candidate"] == candidate]
        summary[candidate] = {
            "psnr": float(np.mean([float(row["psnr"]) for row in rows])),
            "ssim": float(np.mean([float(row["ssim"]) for row in rows])),
            "absdiff": float(np.mean([float(row["absdiff"]) for row in rows])),
        }
        values = summary[candidate]
        print(
            f"{candidate:<20} {len(rows):>5} {values['psnr']:>10.4f} "
            f"{values['ssim']:>10.4f} {values['absdiff']:>10.4f}"
        )

    baseline = summary["ltx16"]
    best_name = max((name for name, _, _ in VARIANTS), key=lambda name: summary[name]["psnr"])
    best = summary[best_name]
    passed = (
        best["psnr"] - baseline["psnr"] >= 0.15
        and best["ssim"] >= baseline["ssim"] - 0.002
        and best["absdiff"] < baseline["absdiff"]
    )
    print()
    print(
        f"best={best_name} dPSNR={best['psnr'] - baseline['psnr']:+.4f} "
        f"dSSIM={best['ssim'] - baseline['ssim']:+.4f} "
        f"dAbsDiff={best['absdiff'] - baseline['absdiff']:+.4f}"
    )
    print(f"gate={'pass' if passed else 'fail'}")
    print(f"output={args.out_csv}")


if __name__ == "__main__":
    main()
