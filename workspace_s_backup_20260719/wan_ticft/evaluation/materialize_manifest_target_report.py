"""Materialize manifest target frames into a simple video report."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2, paths: list[str]):
    frames = []
    for raw in paths:
        frame = read_rgb(cv2, Path(raw))
        if frame is not None:
            frames.append(frame)
    return frames


def load_video_indices(cv2, row: dict[str, Any], indices: list[int]):
    video_path = Path(str(row.get("video_path") or ""))
    if not video_path.exists() or not indices:
        return []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            return []
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def load_targets(cv2, row: dict[str, Any]):
    paths = row.get("frames") or {}
    targets = load_images(cv2, paths.get("target") or [])
    if targets:
        return targets
    indices = row.get("frame_indices") or {}
    return load_video_indices(cv2, row, indices.get("target") or [])


def write_video(cv2, frames, path: Path, fps: float, prepend_guard: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    output_frames = [frames[0], *frames] if prepend_guard else frames
    for frame in output_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--view_type", default=None)
    parser.add_argument("--prepend_guard", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.out_dir / "target"

    rows = [row for row in read_jsonl(args.manifest) if row.get("status", "ok") == "ok"]
    if args.view_type:
        rows = [row for row in rows if row.get("view_type") == args.view_type]
    if args.limit:
        rows = rows[: args.limit]

    report = []
    selected = []
    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        targets = load_targets(cv2, row)
        if not targets:
            print(f"[skip] {sample_id}: no target frames", flush=True)
            continue
        fps = float((row.get("frame_indices") or {}).get("fps") or 30.0)
        path = videos_dir / f"{sample_id}.mp4"
        write_video(cv2, targets, path, fps, args.prepend_guard)
        report.append(
            {
                "sample_id": sample_id,
                "view_type": row.get("view_type", ""),
                "video": str(path),
                "status": "ok",
                "frames": len(targets) + (1 if args.prepend_guard else 0),
                "fps": f"{fps:.8f}",
            }
        )
        selected.append(row)
        print(f"[{index}/{len(rows)}] {sample_id} | ok", flush=True)

    if not report:
        raise RuntimeError("no target videos written")

    with (args.out_dir / "target_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"rows={len(report)}")
    print(f"report={args.out_dir / 'target_report.csv'}")


if __name__ == "__main__":
    main()
