from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import read_report, video_fps
from wan_ticft.evaluation.ltx_detail_donor import all_frames, edge_mask, resize_like, write_video


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def last_condition(cv2, row: dict[str, Any], ref: np.ndarray) -> np.ndarray:
    paths = (row.get("frames") or {}).get("condition") or []
    if not paths:
        raise RuntimeError(f"{row.get('sample_id')}: missing condition frames")
    img = read_rgb(cv2, Path(paths[-1]))
    if img is None:
        raise RuntimeError(f"{row.get('sample_id')}: cannot read condition frame")
    return resize_like(cv2, img, ref)


def masked_detail(cv2, base: np.ndarray, donor: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    donor = resize_like(cv2, donor, base).astype(np.float32)
    low = cv2.GaussianBlur(donor, (0, 0), sigma)
    detail = donor - low
    mask = edge_mask(cv2, base, donor.astype(np.uint8))
    return detail, mask


def compose_detail(
    cv2,
    base: np.ndarray,
    specs: list[tuple[np.ndarray, float, float]],
    lowfreq: tuple[np.ndarray, float, float] | None = None,
) -> np.ndarray:
    out = base.astype(np.float32)
    for donor, alpha, sigma in specs:
        detail, mask = masked_detail(cv2, base, donor, sigma)
        out += alpha * detail * mask
    if lowfreq is not None:
        donor, beta, sigma = lowfreq
        donor = resize_like(cv2, donor, base).astype(np.float32)
        base_low = cv2.GaussianBlur(base.astype(np.float32), (0, 0), sigma)
        donor_low = cv2.GaussianBlur(donor, (0, 0), sigma)
        mask = edge_mask(cv2, base, donor.astype(np.uint8))
        # Low-frequency transfer is deliberately small: it targets residual smear
        # without letting LTX rewrite scene geometry.
        out += beta * (donor_low - base_low) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stronger anti-smear detail donors for BDD val gates.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--traffic_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    cv2, _np = import_cv2_np()
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"][: args.limit]
    base_report = read_report(args.base_report)
    traffic_report = read_report(args.traffic_report)
    labels = [
        "ltxhf_t75",
        "ltxhf_t95",
        "ltxms_t75",
        "real_ltx_t55r25",
        "real_ltx_ms",
        "ltxlf_b12",
        "ltxlf_b20",
        "real_ltx_lf12",
    ]
    reports: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pos, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        base_path = base_report.get(sample_id)
        traffic_path = traffic_report.get(sample_id)
        if base_path is None or traffic_path is None:
            print(f"[{pos}/{len(rows)}] {sample_id} skipped missing input", flush=True)
            continue
        future = max(1, int(row.get("future_frames") or 30))
        fps = video_fps(cv2, base_path) or 30.0
        try:
            base_frames = all_frames(cv2, base_path, future)
            traffic_frames = all_frames(cv2, traffic_path, future)
            real_ref = last_condition(cv2, row, base_frames[0])
        except Exception as exc:  # noqa: BLE001
            print(f"[{pos}/{len(rows)}] {sample_id} bad input {exc!r}", flush=True)
            continue

        frame_sets: dict[str, list[np.ndarray]] = {label: [] for label in labels}
        for idx, base in enumerate(base_frames):
            traffic = traffic_frames[idx]
            real = resize_like(cv2, real_ref, base)
            frame_sets["ltxhf_t75"].append(compose_detail(cv2, base, [(traffic, 0.75, 2.2)]))
            frame_sets["ltxhf_t95"].append(compose_detail(cv2, base, [(traffic, 0.95, 2.2)]))
            frame_sets["ltxms_t75"].append(
                compose_detail(cv2, base, [(traffic, 0.60, 2.2), (traffic, 0.25, 5.0)], (traffic, 0.05, 7.0))
            )
            frame_sets["real_ltx_t55r25"].append(compose_detail(cv2, base, [(traffic, 0.55, 2.2), (real, 0.25, 2.0)]))
            frame_sets["real_ltx_ms"].append(
                compose_detail(cv2, base, [(traffic, 0.55, 2.2), (real, 0.20, 2.0), (traffic, 0.20, 5.0)], (traffic, 0.04, 7.0))
            )
            frame_sets["ltxlf_b12"].append(
                compose_detail(cv2, base, [(traffic, 0.70, 2.2), (traffic, 0.25, 5.0)], (traffic, 0.12, 7.0))
            )
            frame_sets["ltxlf_b20"].append(
                compose_detail(cv2, base, [(traffic, 0.70, 2.2), (traffic, 0.25, 5.0)], (traffic, 0.20, 7.0))
            )
            frame_sets["real_ltx_lf12"].append(
                compose_detail(cv2, base, [(traffic, 0.55, 2.2), (real, 0.25, 2.0), (traffic, 0.20, 5.0)], (traffic, 0.12, 7.0))
            )

        for label, frames in frame_sets.items():
            out_path = args.out_dir / label / "videos" / f"{sample_id}.mp4"
            write_video(cv2, out_path, frames, fps)
            reports[label].append(
                {
                    "sample_id": sample_id,
                    "video": str(out_path),
                    "status": "ok",
                    "source_fps": f"{fps:.6f}",
                    "future_frames": len(frames),
                    "notes": label,
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
