from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from .console import kv, section
from .frame_io import import_cv2_np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def discover_videos(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return [path]
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_one(
    cv2,
    video_path: Path,
    out_dir: Path,
    every_n: int,
    start_index: int,
    jpg: bool,
    resize_width: int,
    resize_height: int,
    clean: bool,
) -> dict[str, Any]:
    sample_dir = out_dir / video_path.stem
    if clean:
        clean_dir(sample_dir)
    else:
        sample_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video": str(video_path),
            "out_dir": str(sample_dir),
            "status": "failed",
            "frames_in_video": 0,
            "frames_written": 0,
            "fps": "",
            "width": "",
            "height": "",
            "notes": "cannot open video",
        }

    frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    written = 0
    read_idx = 0
    failed_writes = 0
    ext = ".jpg" if jpg else ".png"

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if read_idx % every_n == 0:
            if resize_width > 0 and resize_height > 0:
                frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            out_path = sample_dir / f"{start_index + written}{ext}"
            if not cv2.imwrite(str(out_path), frame):
                failed_writes += 1
            else:
                written += 1

        read_idx += 1

    cap.release()

    return {
        "video": str(video_path),
        "out_dir": str(sample_dir),
        "status": "ok" if failed_writes == 0 and written > 0 else "partial",
        "frames_in_video": frames_in_video,
        "frames_written": written,
        "fps": fps,
        "width": width,
        "height": height,
        "notes": "" if failed_writes == 0 else f"failed_writes={failed_writes}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from one video or a directory of predicted videos.")
    parser.add_argument("--videos", required=True, type=Path, help="MP4 file or directory containing predicted videos.")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output root. Each video gets one subfolder.")
    parser.add_argument("--report_csv", type=Path, default=None)
    parser.add_argument("--every_n", type=int, default=1, help="Extract every Nth frame. Default extracts all frames.")
    parser.add_argument("--start_index", type=int, default=0, help="First output frame index.")
    parser.add_argument("--jpg", action="store_true", help="Write JPG instead of PNG for smaller debug output.")
    parser.add_argument("--resize_width", type=int, default=0)
    parser.add_argument("--resize_height", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clean", action="store_true", help="Delete each sample output folder before writing.")
    args = parser.parse_args()

    if args.every_n <= 0:
        raise SystemExit("--every_n must be >= 1")

    cv2, _ = import_cv2_np()
    videos = discover_videos(args.videos)
    if args.limit > 0:
        videos = videos[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_csv = args.report_csv or (args.out_dir / "extract_video_frames_report.csv")

    section("EXTRACT VIDEO FRAMES")
    kv("videos", args.videos)
    kv("out_dir", args.out_dir)
    kv("video files", len(videos))
    kv("every_n", args.every_n)
    kv("format", "jpg" if args.jpg else "png")

    rows = []
    for video_path in videos:
        row = extract_one(
            cv2=cv2,
            video_path=video_path,
            out_dir=args.out_dir,
            every_n=args.every_n,
            start_index=args.start_index,
            jpg=args.jpg,
            resize_width=args.resize_width,
            resize_height=args.resize_height,
            clean=args.clean,
        )
        rows.append(row)

    with report_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["video", "out_dir", "status", "frames_in_video", "frames_written", "fps", "width", "height", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    section("RESULT")
    kv("report_csv", report_csv)
    kv("videos processed", len(rows))
    kv("frames written", sum(int(r["frames_written"]) for r in rows))
    kv("by_status", dict((s, sum(1 for r in rows if r["status"] == s)) for s in sorted({r["status"] for r in rows})))


if __name__ == "__main__":
    main()
