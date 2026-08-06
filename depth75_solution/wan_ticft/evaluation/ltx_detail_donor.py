from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from baseline.frame_io import import_cv2_np
from wan_ticft.evaluation.evaluate_video_reports import read_report, read_video_rgb_frames, video_fps


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def all_frames(cv2, path: Path, target_count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while len(frames) < target_count:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"cannot read video: {path}")
    while len(frames) < target_count:
        frames.append(frames[-1].copy())
    return frames[:target_count]


def resize_like(cv2, image: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if image.shape[:2] == ref.shape[:2]:
        return image
    return cv2.resize(image, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)


def edge_mask(cv2, base: np.ndarray, donor: np.ndarray) -> np.ndarray:
    base_gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    donor_gray = cv2.cvtColor(donor, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bx = cv2.Sobel(base_gray, cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(base_gray, cv2.CV_32F, 0, 1, ksize=3)
    dx = cv2.Sobel(donor_gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(donor_gray, cv2.CV_32F, 0, 1, ksize=3)
    bg = cv2.magnitude(bx, by)
    dg = cv2.magnitude(dx, dy)
    # Prefer donor detail where both videos agree that a real edge/texture exists.
    common = np.minimum(bg, dg) / (np.maximum(bg, dg) + 12.0)
    textured = np.clip((bg + dg - 10.0) / 45.0, 0.0, 1.0)
    mask = np.clip(0.15 + 0.85 * common * textured, 0.15, 1.0)
    return cv2.GaussianBlur(mask, (0, 0), 1.2)[..., None]


def transfer_detail(cv2, base: np.ndarray, donors: list[np.ndarray], alpha: float, sigma: float) -> np.ndarray:
    base_f = base.astype(np.float32)
    detail_acc = np.zeros_like(base_f)
    mask_acc = np.zeros((*base.shape[:2], 1), dtype=np.float32)
    for donor in donors:
        donor = resize_like(cv2, donor, base).astype(np.float32)
        low = cv2.GaussianBlur(donor, (0, 0), sigma)
        detail = donor - low
        mask = edge_mask(cv2, base, donor.astype(np.uint8))
        detail_acc += detail * mask
        mask_acc += mask
    detail_avg = detail_acc / np.maximum(mask_acc, 1e-3)
    out = base_f + alpha * detail_avg * np.clip(mask_acc / max(len(donors), 1), 0.0, 1.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def write_video(cv2, path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Borrow only high-frequency detail from LTX videos.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--traffic_report", type=Path, required=True)
    parser.add_argument("--dashcam_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=2.2)
    args = parser.parse_args()

    cv2, _np = import_cv2_np()
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"][: args.limit]
    base_report = read_report(args.base_report)
    traffic_report = read_report(args.traffic_report)
    dashcam_report = read_report(args.dashcam_report)
    variants = {
        "ltxhf_t35": (0.35, ("traffic",)),
        "ltxhf_t55": (0.55, ("traffic",)),
        "ltxhf_dual45": (0.45, ("traffic", "dashcam")),
    }
    reports: dict[str, list[dict[str, Any]]] = {label: [] for label in variants}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pos, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        base_path = base_report.get(sample_id)
        traffic_path = traffic_report.get(sample_id)
        dashcam_path = dashcam_report.get(sample_id)
        if base_path is None or traffic_path is None or dashcam_path is None:
            print(f"[{pos}/{len(rows)}] {sample_id} skipped missing input", flush=True)
            continue
        future = max(1, int(row.get("future_frames") or 30))
        fps = video_fps(cv2, base_path) or 30.0
        try:
            base_frames = all_frames(cv2, base_path, future)
            donor_frames = {
                "traffic": all_frames(cv2, traffic_path, future),
                "dashcam": all_frames(cv2, dashcam_path, future),
            }
        except Exception as exc:  # noqa: BLE001
            print(f"[{pos}/{len(rows)}] {sample_id} bad input {exc!r}", flush=True)
            continue

        for label, (alpha, donor_names) in variants.items():
            out_path = args.out_dir / label / "videos" / f"{sample_id}.mp4"
            frames = [
                transfer_detail(
                    cv2,
                    base,
                    [donor_frames[name][idx] for name in donor_names],
                    alpha=alpha,
                    sigma=args.sigma,
                )
                for idx, base in enumerate(base_frames)
            ]
            write_video(cv2, out_path, frames, fps)
            reports[label].append(
                {
                    "sample_id": sample_id,
                    "video": str(out_path),
                    "status": "ok",
                    "source_fps": f"{fps:.6f}",
                    "future_frames": len(frames),
                    "notes": f"alpha={alpha};donors={'+'.join(donor_names)};sigma={args.sigma}",
                }
            )
        print(f"[{pos}/{len(rows)}] {sample_id} ok", flush=True)

    for label, report_rows in reports.items():
        report_path = args.out_dir / label / "report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "video", "status", "source_fps", "future_frames", "notes"],
            )
            writer.writeheader()
            writer.writerows(report_rows)
    print(f"output={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
