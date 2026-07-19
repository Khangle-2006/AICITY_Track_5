"""Materialize repeat-last videos aligned to an existing reference report."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2

from baseline.frame_io import read_rgb
from wan_ticft.evaluation.evaluate_video_reports import read_report, video_fps


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_video(frame, path: Path, fps: float, frame_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for _ in range(frame_count):
        writer.write(bgr)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = read_report(args.reference_report)
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == args.view_type and row["sample_id"] in reference]
    if args.limit is not None:
        rows = rows[: args.limit]

    report_rows = []
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        condition_paths = (row.get("frames") or {}).get("condition") or []
        if not condition_paths:
            print(f"[{position}/{len(rows)}] {sample_id} | skip no condition", flush=True)
            continue
        anchor = read_rgb(cv2, Path(condition_paths[-1]))
        reference_video = reference[sample_id]
        cap = cv2.VideoCapture(str(reference_video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, first_bgr = cap.read()
        cap.release()
        if anchor is None or frame_count <= 0 or not ok or first_bgr is None:
            print(f"[{position}/{len(rows)}] {sample_id} | skip unreadable", flush=True)
            continue
        height, width = first_bgr.shape[:2]
        if anchor.shape[:2] != (height, width):
            anchor = cv2.resize(anchor, (width, height), interpolation=cv2.INTER_AREA)
        fps = video_fps(cv2, reference_video) or 30.0
        video_path = args.out_dir / "videos" / f"{sample_id}.avi"
        write_video(anchor, video_path, fps, frame_count)
        report_rows.append(
            {"sample_id": sample_id, "video": str(video_path), "status": "ok", "future_frames": frame_count}
        )
        print(f"[{position}/{len(rows)}] {sample_id} | frames={frame_count}", flush=True)

    if not report_rows:
        raise RuntimeError("no repeat-last videos written")
    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
