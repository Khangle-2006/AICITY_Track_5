"""Measure the ceiling of spatially compositing LTX and pixel-preserving predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_bdd_ego_motion import region_predictions, score_sequence
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    source_fps,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_images(cv2, paths: list[str]):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def target_offsets(row: dict, count: int) -> list[int]:
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    target = indices.get("target") or []
    if not condition or len(target) != count:
        return list(range(1, count + 1))
    last = int(condition[-1])
    return [max(1, int(index) - last) for index in target]


def resize_like(cv2, image, target):
    if image.shape[:2] == target.shape[:2]:
        return image
    return cv2.resize(image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)


def block_oracle(np, first, second, target, block: int):
    output = first.copy()
    height, width = target.shape[:2]
    for y in range(0, height, block):
        for x in range(0, width, block):
            ys = slice(y, min(height, y + block))
            xs = slice(x, min(width, x + block))
            target_patch = target[ys, xs].astype(np.float32)
            first_error = np.mean(np.abs(first[ys, xs].astype(np.float32) - target_patch))
            second_error = np.mean(np.abs(second[ys, xs].astype(np.float32) - target_patch))
            if second_error < first_error:
                output[ys, xs] = second[ys, xs]
    return output


def pixel_oracle(np, first, second, target):
    first_error = np.mean(np.abs(first.astype(np.float32) - target.astype(np.float32)), axis=2)
    second_error = np.mean(np.abs(second.astype(np.float32) - target.astype(np.float32)), axis=2)
    return np.where((second_error < first_error)[:, :, None], second, first)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--estimate_width", type=int, default=640)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    reports = read_report(args.ltx_report)
    rows = []
    for position, row in enumerate(read_jsonl(args.manifest), start=1):
        sample_id = row["sample_id"]
        video = reports.get(sample_id)
        paths = row.get("frames") or {}
        conditions = load_images(cv2, paths.get("condition") or [])
        targets = load_images(cv2, paths.get("target") or [])
        if video is None or len(conditions) < 2 or not targets:
            print(f"[{position}] {sample_id} | skipped", flush=True)
            continue

        condition_indices = (row.get("frame_indices") or {}).get("condition") or []
        history_gap = (
            max(1, int(condition_indices[-1]) - int(condition_indices[-2]))
            if len(condition_indices) >= 2
            else 1
        )
        offsets = target_offsets(row, len(targets))
        region = region_predictions(
            cv2, np, conditions[-2], conditions[-1], offsets, history_gap, args.estimate_width
        )
        generated_fps = video_fps(cv2, video)
        mapped = map_target_offsets(row, generated_fps)
        frame_map = read_video_rgb_frames(cv2, video, mapped)
        ltx = [frame_map[index] for index in mapped if index in frame_map]
        if len(ltx) != len(targets):
            print(f"[{position}] {sample_id} | incomplete LTX", flush=True)
            continue

        candidates = {"region": [], "ltx16": [], "frame_oracle": [], "block64_oracle": [], "block32_oracle": [], "pixel_oracle": []}
        for region_frame, ltx_frame, target in zip(region, ltx, targets):
            region_frame = resize_like(cv2, region_frame, target)
            ltx_frame = resize_like(cv2, ltx_frame, target)
            candidates["region"].append(region_frame)
            candidates["ltx16"].append(ltx_frame)
            region_error = np.mean(np.abs(region_frame.astype(np.float32) - target.astype(np.float32)))
            ltx_error = np.mean(np.abs(ltx_frame.astype(np.float32) - target.astype(np.float32)))
            candidates["frame_oracle"].append(ltx_frame if ltx_error < region_error else region_frame)
            candidates["block64_oracle"].append(block_oracle(np, region_frame, ltx_frame, target, 64))
            candidates["block32_oracle"].append(block_oracle(np, region_frame, ltx_frame, target, 32))
            candidates["pixel_oracle"].append(pixel_oracle(np, region_frame, ltx_frame, target))

        for candidate, predictions in candidates.items():
            p, s, d = score_sequence(cv2, np, predictions, targets)
            rows.append(
                {
                    "sample_id": sample_id,
                    "phase": row.get("phase", ""),
                    "candidate": candidate,
                    "psnr": f"{p:.6f}",
                    "ssim": f"{s:.6f}",
                    "absdiff": f"{d:.6f}",
                    "source_fps": f"{source_fps(row):.6f}",
                }
            )
        print(f"[{position}] {sample_id} | ok", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'Candidate':<20} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 60)
    for candidate in ["region", "ltx16", "frame_oracle", "block64_oracle", "block32_oracle", "pixel_oracle"]:
        items = [row for row in rows if row["candidate"] == candidate]
        print(
            f"{candidate:<20} {len(items):>5} "
            f"{np.mean([float(x['psnr']) for x in items]):>10.4f} "
            f"{np.mean([float(x['ssim']) for x in items]):>10.4f} "
            f"{np.mean([float(x['absdiff']) for x in items]):>10.4f}"
        )
    print(f"output {args.out_csv}")


if __name__ == "__main__":
    main()
