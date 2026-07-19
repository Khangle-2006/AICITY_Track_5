"""Gate motion-aligned last-frame texture rescue on BDD validation videos.

This targets global afterimage/smear in the generated background. It aligns the
last real condition frame to each generated future frame, then transfers only
luma detail where the alignment is confident. The output remains a video report
so it can be scored with the same REAL-FID gate as other BDD candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resize_like(cv2, image, reference):
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)


def dis_flow(cv2, source_gray, target_gray):
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setUseSpatialPropagation(True)
    return estimator.calc(source_gray, target_gray, None)


def remap(cv2, np, image, flow):
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return cv2.remap(
        image,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )


def align_anchor(cv2, np, base, anchor, fb_scale: float, photo_scale: float):
    base_gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    anchor_gray = cv2.cvtColor(anchor, cv2.COLOR_RGB2GRAY)
    flow_base_to_anchor = dis_flow(cv2, base_gray, anchor_gray)
    flow_anchor_to_base = dis_flow(cv2, anchor_gray, base_gray)
    aligned = remap(cv2, np, anchor, flow_base_to_anchor)

    reverse_at_base = remap(cv2, np, flow_anchor_to_base, flow_base_to_anchor)
    fb_error = np.linalg.norm(flow_base_to_anchor + reverse_at_base, axis=2)
    fb_conf = np.exp(-fb_error / max(fb_scale, 1e-6))

    base_low = cv2.GaussianBlur(base.astype(np.float32), (0, 0), 3.0)
    aligned_low = cv2.GaussianBlur(aligned.astype(np.float32), (0, 0), 3.0)
    photo_error = np.mean(np.abs(base_low - aligned_low), axis=2)
    photo_conf = np.exp(-photo_error / max(photo_scale, 1e-6))
    confidence = np.clip(fb_conf * photo_conf, 0.0, 1.0).astype(np.float32)
    confidence = cv2.GaussianBlur(confidence, (0, 0), 1.2)
    return aligned, np.clip(confidence, 0.0, 1.0)


def bandpass(cv2, image, sigma_low: float, sigma_high: float):
    low = cv2.GaussianBlur(image, (0, 0), sigma_low)
    high = cv2.GaussianBlur(image, (0, 0), sigma_high)
    return high - low


def rescue_from_aligned(cv2, np, base, aligned, confidence, variant: str):
    conf = confidence.astype(np.float32)

    base_ycc = cv2.cvtColor(base, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    aligned_ycc = cv2.cvtColor(aligned, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    base_y = base_ycc[..., 0]
    anchor_y = aligned_ycc[..., 0]

    base_high = base_y - cv2.GaussianBlur(base_y, (0, 0), 1.6)
    anchor_high = anchor_y - cv2.GaussianBlur(anchor_y, (0, 0), 1.6)
    high_delta = np.clip(anchor_high - base_high, -22.0, 22.0)

    base_mid = bandpass(cv2, base_y, 1.8, 7.0)
    anchor_mid = bandpass(cv2, anchor_y, 1.8, 7.0)
    mid_delta = np.clip(anchor_mid - base_mid, -18.0, 18.0)

    base_low = cv2.GaussianBlur(base_y, (0, 0), 7.0)
    anchor_low = cv2.GaussianBlur(anchor_y, (0, 0), 7.0)
    low_delta = np.clip(anchor_low - base_low, -14.0, 14.0)

    if variant == "yhf_a25":
        out_y = base_y + 0.25 * conf * high_delta
    elif variant == "ymid_a20":
        out_y = base_y + conf * (0.22 * high_delta + 0.16 * mid_delta)
    elif variant == "ysmear_a12":
        out_y = base_y + conf * (0.20 * high_delta + 0.12 * mid_delta + 0.08 * low_delta)
    elif variant == "yfull_a08":
        out_y = base_y * (1.0 - 0.08 * conf) + anchor_y * (0.08 * conf)
    else:
        raise ValueError(f"unknown variant: {variant}")

    base_ycc[..., 0] = np.clip(out_y, 0.0, 255.0)
    return cv2.cvtColor(base_ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def read_last_condition(cv2, row: dict):
    paths = (row.get("frames") or {}).get("condition") or []
    return read_rgb(cv2, Path(paths[-1])) if paths else None


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--variants", default="yhf_a25,ymid_a20,ysmear_a12,yfull_a08")
    parser.add_argument("--fb_scale", type=float, default=2.5)
    parser.add_argument("--photo_scale", type=float, default=22.0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == args.view_type][: args.limit]
    base_report = read_report(args.base_report)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    report_rows = {variant: [] for variant in variants}

    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        base_video = base_report.get(sample_id)
        anchor = read_last_condition(cv2, row)
        if base_video is None or anchor is None:
            print(f"[{position}/{len(rows)}] {sample_id} | skip", flush=True)
            continue

        fps = video_fps(cv2, base_video) or 30.0
        offsets = sorted(set(map_target_offsets(row, fps)))
        max_offset = max(offsets)
        timeline_indices = list(range(max_offset + 1))
        base_map = read_video_rgb_frames(cv2, base_video, timeline_indices)
        if len(base_map) != len(timeline_indices):
            print(f"[{position}/{len(rows)}] {sample_id} | skip frames {len(base_map)}/{len(timeline_indices)}", flush=True)
            continue

        output_by_variant = {
            variant: [base_map[index].copy() for index in timeline_indices]
            for variant in variants
        }
        confidences = []
        for index in offsets:
            base = base_map[index]
            aligned, confidence = align_anchor(
                cv2,
                np,
                base,
                resize_like(cv2, anchor, base),
                args.fb_scale,
                args.photo_scale,
            )
            confidences.append(float(confidence.mean()))
            for variant in variants:
                output_by_variant[variant][index] = rescue_from_aligned(cv2, np, base, aligned, confidence, variant)

        mean_conf = float(np.mean(confidences))
        summaries = []
        for variant in variants:
            video_path = args.out_root / variant / "videos" / f"{sample_id}.avi"
            write_video(cv2, output_by_variant[variant], video_path, fps)
            report_rows[variant].append(
                {
                    "sample_id": sample_id,
                    "video": str(video_path),
                    "status": "ok",
                    "future_frames": len(output_by_variant[variant]),
                    "mean_confidence": f"{mean_conf:.6f}",
                }
            )
            summaries.append(f"{variant}:{mean_conf:.3f}")
        print(f"[{position}/{len(rows)}] {sample_id} | {' '.join(summaries)}", flush=True)

    for variant, rows_out in report_rows.items():
        if not rows_out:
            continue
        report_path = args.out_root / variant / "report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
    print(f"output={args.out_root}")


if __name__ == "__main__":
    main()
