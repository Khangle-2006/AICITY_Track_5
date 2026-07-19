"""Materialize a video report into compact target-aligned future-frame videos."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np
from wan_ticft.evaluation.evaluate_video_reports import (
    load_targets,
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    target_offsets,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def infer_view_type(row: dict[str, str], manifest_row: dict) -> str:
    return row.get("view_type") or manifest_row.get("view_type", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--target_only", action="store_true")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--offset_mode", choices=["frame", "timestamp", "sequential"], default="frame")
    parser.add_argument("--sequential_start", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.target_only and args.report is None:
        raise ValueError("--report is required unless --target_only is used")
    report = {} if args.target_only else read_report(args.report)
    row_meta = {}
    if args.report is not None:
        with args.report.open("r", encoding="utf-8", newline="") as handle:
            row_meta = {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("status") == "ok"}

    out_rows = []
    for position, row in enumerate(read_jsonl(args.manifest), start=1):
        sid = row["sample_id"]
        targets = load_targets(cv2, row)
        if not targets:
            continue
        if args.target_only:
            frames = targets
        else:
            video = report.get(sid)
            if video is None:
                continue
            fps = video_fps(cv2, video) or 30.0
            if args.offset_mode == "timestamp":
                offsets = map_target_offsets(row, fps)
            elif args.offset_mode == "sequential":
                offsets = list(range(args.sequential_start, args.sequential_start + len(targets)))
            else:
                offsets = target_offsets(row)
            frame_map = read_video_rgb_frames(cv2, video, offsets)
            frames = [frame_map[index] for index in offsets if index in frame_map]
            if len(frames) != len(targets):
                print(f"[{position}] {sid} | skip read={len(frames)}/{len(targets)}", flush=True)
                continue
        out_video = args.out_dir / "videos" / f"{sid}.avi"
        write_video(cv2, frames, out_video, 30.0)
        meta = row_meta.get(sid, {})
        out_rows.append(
            {
                "sample_id": sid,
                "view_type": infer_view_type(meta, row),
                "video": str(out_video),
                "status": "ok",
                "future_frames": len(frames),
                "fps": "30.0",
            }
        )
        print(f"[{position}] {sid} | ok", flush=True)

    if not out_rows:
        raise RuntimeError("no target-aligned videos written")
    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_rows[0].keys())
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"rows={len(out_rows)}")
    print(f"output={args.out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
