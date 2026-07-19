"""Measure the GT oracle ceiling for mixing MF and LTX validation videos.

This is not a deployable method: it uses target frames to choose between two
existing predictions. The goal is to decide whether an LTX selector/compositor
has enough upside to justify building a non-oracle version.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.evaluation.evaluate_video_reports import (
    load_targets,
    map_target_offsets,
    read_manifest,
    read_report,
    read_video_rgb_frames,
    resize_like,
    target_offsets,
    video_fps,
)


def read_candidate_frames(cv2, row: dict[str, Any], video: Path, offset_mode: str) -> list[Any]:
    gen_fps = video_fps(cv2, video)
    offsets = map_target_offsets(row, gen_fps) if offset_mode == "timestamp" else target_offsets(row)
    frame_map = read_video_rgb_frames(cv2, video, offsets)
    return [frame_map[index] for index in offsets if index in frame_map]


def score_frames(cv2, np, preds: list[Any], targets: list[Any]) -> tuple[float, float, float]:
    values = []
    for pred, target in zip(preds, targets):
        pred = resize_like(cv2, pred, target)
        values.append(
            (
                psnr(cv2, np, pred, target),
                ssim_gray(cv2, np, pred, target),
                mean_absdiff(np, pred, target),
            )
        )
    return tuple(float(np.mean([item[index] for item in values])) for index in range(3))


def frame_oracle(np, mf, ltx, target):
    mf_err = np.mean(np.abs(mf.astype(np.float32) - target.astype(np.float32)))
    ltx_err = np.mean(np.abs(ltx.astype(np.float32) - target.astype(np.float32)))
    return (ltx if ltx_err < mf_err else mf), float(ltx_err < mf_err)


def block_oracle(cv2, np, mf, ltx, target, block: int):
    mf = resize_like(cv2, mf, target)
    ltx = resize_like(cv2, ltx, target)
    height, width = target.shape[:2]
    output = mf.copy()
    ltx_pixels = 0
    for y0 in range(0, height, block):
        y1 = min(height, y0 + block)
        for x0 in range(0, width, block):
            x1 = min(width, x0 + block)
            mf_patch = mf[y0:y1, x0:x1].astype(np.float32)
            ltx_patch = ltx[y0:y1, x0:x1].astype(np.float32)
            tgt_patch = target[y0:y1, x0:x1].astype(np.float32)
            mf_err = np.mean(np.abs(mf_patch - tgt_patch))
            ltx_err = np.mean(np.abs(ltx_patch - tgt_patch))
            if ltx_err < mf_err:
                output[y0:y1, x0:x1] = ltx[y0:y1, x0:x1]
                ltx_pixels += (y1 - y0) * (x1 - x0)
    return output, float(ltx_pixels / (height * width))


def pixel_oracle(cv2, np, mf, ltx, target):
    mf = resize_like(cv2, mf, target)
    ltx = resize_like(cv2, ltx, target)
    mf_err = np.mean(np.abs(mf.astype(np.float32) - target.astype(np.float32)), axis=2)
    ltx_err = np.mean(np.abs(ltx.astype(np.float32) - target.astype(np.float32)), axis=2)
    mask = ltx_err < mf_err
    output = mf.copy()
    output[mask] = ltx[mask]
    return output, float(np.mean(mask))


def summarize(np, rows: list[dict[str, Any]]) -> None:
    candidates = list(dict.fromkeys(row["candidate"] for row in rows))
    print(f"{'Candidate':<20} {'N':>5} {'LTX%':>8} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 70)
    for candidate in candidates:
        items = [row for row in rows if row["candidate"] == candidate]
        print(
            f"{candidate:<20} {len(items):>5} "
            f"{100.0 * np.mean([float(x['ltx_fraction']) for x in items]):>8.2f} "
            f"{np.mean([float(x['psnr']) for x in items]):>10.4f} "
            f"{np.mean([float(x['ssim']) for x in items]):>10.4f} "
            f"{np.mean([float(x['absdiff']) for x in items]):>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mf_report", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--offset_mode", choices=["index", "timestamp"], default="timestamp")
    parser.add_argument("--blocks", type=int, nargs="+", default=[64, 32, 16])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    manifest = read_manifest(args.manifest)
    mf_report = read_report(args.mf_report)
    ltx_report = read_report(args.ltx_report)
    rows_out: list[dict[str, Any]] = []
    processed = 0

    for index, row in enumerate(manifest, start=1):
        sample_id = str(row.get("sample_id", ""))
        if args.limit and processed >= args.limit:
            break
        mf_video = mf_report.get(sample_id)
        ltx_video = ltx_report.get(sample_id)
        if mf_video is None or ltx_video is None:
            continue
        targets = load_targets(cv2, row)
        mf_frames = read_candidate_frames(cv2, row, mf_video, args.offset_mode)
        ltx_frames = read_candidate_frames(cv2, row, ltx_video, args.offset_mode)
        if not targets or len(mf_frames) != len(targets) or len(ltx_frames) != len(targets):
            print(
                f"[{index}] {sample_id} skipped targets={len(targets)} "
                f"mf={len(mf_frames)} ltx={len(ltx_frames)}",
                flush=True,
            )
            continue

        candidates: dict[str, list[Any]] = {"mf": [], "ltx": [], "frame_oracle": []}
        candidates.update({f"block{block}_oracle": [] for block in args.blocks})
        candidates["pixel_oracle"] = []
        fractions: dict[str, list[float]] = {key: [] for key in candidates}
        for target, mf, ltx in zip(targets, mf_frames, ltx_frames):
            mf = resize_like(cv2, mf, target)
            ltx = resize_like(cv2, ltx, target)
            candidates["mf"].append(mf)
            candidates["ltx"].append(ltx)
            fractions["mf"].append(0.0)
            fractions["ltx"].append(1.0)

            pred, frac = frame_oracle(np, mf, ltx, target)
            candidates["frame_oracle"].append(pred)
            fractions["frame_oracle"].append(frac)
            for block in args.blocks:
                label = f"block{block}_oracle"
                pred, frac = block_oracle(cv2, np, mf, ltx, target, block)
                candidates[label].append(pred)
                fractions[label].append(frac)
            pred, frac = pixel_oracle(cv2, np, mf, ltx, target)
            candidates["pixel_oracle"].append(pred)
            fractions["pixel_oracle"].append(frac)

        for candidate, preds in candidates.items():
            p, s, d = score_frames(cv2, np, preds, targets)
            rows_out.append(
                {
                    "sample_id": sample_id,
                    "phase": row.get("phase", ""),
                    "candidate": candidate,
                    "target_count": len(targets),
                    "ltx_fraction": f"{np.mean(fractions[candidate]):.6f}",
                    "psnr": f"{p:.6f}",
                    "ssim": f"{s:.6f}",
                    "absdiff": f"{d:.6f}",
                }
            )
        processed += 1
        print(f"[{processed}] {sample_id} ok", flush=True)

    if not rows_out:
        raise RuntimeError("no samples evaluated")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    summarize(np, rows_out)
    print(f"output {args.out_csv}")


if __name__ == "__main__":
    main()
