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


def condition_ref(cv2, row: dict[str, Any], ref: np.ndarray) -> np.ndarray:
    paths = (row.get("frames") or {}).get("condition") or []
    if not paths:
        raise RuntimeError("missing condition frames")
    img = read_rgb(cv2, Path(paths[-1]))
    if img is None:
        raise RuntimeError(f"cannot read {paths[-1]}")
    return resize_like(cv2, img, ref)


def compose(cv2, base: np.ndarray, specs: list[tuple[np.ndarray, float, float]]) -> np.ndarray:
    out = base.astype(np.float32)
    for donor, alpha, sigma in specs:
        donor = resize_like(cv2, donor, base).astype(np.float32)
        low = cv2.GaussianBlur(donor, (0, 0), sigma)
        detail = donor - low
        mask = edge_mask(cv2, base, donor.astype(np.uint8))
        out += alpha * detail * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep stronger real-frame high-frequency donors.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--traffic_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    cv2, _np = import_cv2_np()
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"][: args.limit]
    base_report = read_report(args.base_report)
    traffic_report = read_report(args.traffic_report)
    variants = {
        "real_ltx_r40": [(0.55, "traffic", 2.2), (0.40, "real", 2.0)],
        "real_ltx_r60": [(0.45, "traffic", 2.2), (0.60, "real", 2.0)],
        "realonly_r50": [(0.50, "real", 2.0)],
        "real_ltx_t75r35": [(0.75, "traffic", 2.2), (0.35, "real", 2.0)],
    }
    reports: dict[str, list[dict[str, Any]]] = {label: [] for label in variants}
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
            real_ref = condition_ref(cv2, row, base_frames[0])
        except Exception as exc:  # noqa: BLE001
            print(f"[{pos}/{len(rows)}] {sample_id} bad input {exc!r}", flush=True)
            continue

        outputs = {label: [] for label in variants}
        for idx, base in enumerate(base_frames):
            donors = {"traffic": traffic_frames[idx], "real": resize_like(cv2, real_ref, base)}
            for label, recipe in variants.items():
                outputs[label].append(compose(cv2, base, [(donors[name], alpha, sigma) for alpha, name, sigma in recipe]))
        for label, frames in outputs.items():
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

    for label, rows_out in reports.items():
        report_path = args.out_dir / label / "report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "video", "status", "source_fps", "future_frames", "notes"],
            )
            writer.writeheader()
            writer.writerows(rows_out)
    print(f"output={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
