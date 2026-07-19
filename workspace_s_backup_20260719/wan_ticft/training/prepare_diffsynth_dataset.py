"""Prepare DiffSynth-Studio Wan training data from a TIC-FT manifest.

DiffSynth's Wan trainer expects a metadata file with a video path and prompt.
For TI2V training with ``--extra_inputs input_image``, its trainer uses the
first frame of each video as the input image. To match our inference setup, we
build each clip as:

    [last condition frame] + target frames + repeated final target frame

and pad/truncate to a Wan-compatible ``4n+1`` frame count.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_kv(key: str, value: object, width: int = 42) -> None:
    print(f"{key:<{width}} {value}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_name(value: str) -> str:
    import re

    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "sample"


def ensure_4n_plus_1(num_frames: int) -> int:
    if num_frames < 5:
        return 5
    rem = num_frames % 4
    if rem == 1:
        return num_frames
    return num_frames + ((1 - rem) % 4)


def build_clip_paths(row: dict[str, Any], num_frames: int) -> tuple[list[Path], str]:
    frames = row.get("frames") or {}
    condition = [Path(p) for p in frames.get("condition") or []]
    target = [Path(p) for p in frames.get("target") or []]
    if not condition:
        return [], "missing condition frames"
    if not target:
        return [], "missing target frames"

    clip = [condition[-1]] + target
    while len(clip) < num_frames:
        clip.append(clip[-1])
    clip = clip[:num_frames]

    missing = [str(p) for p in clip if not p.exists()]
    if missing:
        return [], f"missing frame files: {missing[:3]}"
    return clip, ""


def write_clip(
    *,
    row: dict[str, Any],
    out_video: Path,
    num_frames: int,
    width: int,
    height: int,
    fps: float,
) -> str:
    try:
        import cv2
    except Exception as exc:
        return f"opencv import failed: {exc}"

    clip, note = build_clip_paths(row, num_frames)
    if note:
        return note

    out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        return f"could not open writer: {out_video}"

    try:
        for path in clip:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                return f"failed reading frame: {path}"
            h, w = frame.shape[:2]
            if w != width or h != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    if not out_video.exists() or out_video.stat().st_size == 0:
        return f"writer produced empty file: {out_video}"
    return ""


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt", "sample_id", "view_type", "dataset_part", "phase"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = ["sample_id", "status", "video", "prompt_words", "view_type", "dataset_part", "phase", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_frames", type=int, default=13)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no_clean", action="store_false", dest="clean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    args.num_frames = ensure_4n_plus_1(args.num_frames)

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.out_dir / "videos"
    metadata_csv = args.out_dir / "metadata.csv"
    report_csv = args.out_dir / "prepare_report.csv"

    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    print_header("PREPARE DIFFSYNTH WAN DATASET")
    print_kv("manifest", args.manifest)
    print_kv("out_dir", args.out_dir)
    print_kv("rows", len(rows))
    print_kv("num_frames", args.num_frames)
    print_kv("size", f"{args.width}x{args.height}")

    metadata_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, Any]] = []
    ok = 0
    bad = 0
    for idx, row in enumerate(rows):
        sample_id = safe_name(str(row.get("sample_id") or f"sample_{idx:06d}"))
        out_video = video_dir / f"{idx:06d}_{sample_id}.mp4"
        note = write_clip(
            row=row,
            out_video=out_video,
            num_frames=args.num_frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        status = "ok" if not note else "bad"
        if status == "ok":
            ok += 1
            metadata_rows.append(
                {
                    "video": str(out_video.relative_to(args.out_dir)),
                    "prompt": str(row.get("prompt") or ""),
                    "sample_id": str(row.get("sample_id") or ""),
                    "view_type": str(row.get("view_type") or ""),
                    "dataset_part": str(row.get("dataset_part") or ""),
                    "phase": str(row.get("phase") or ""),
                }
            )
        else:
            bad += 1
        report_rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "status": status,
                "video": str(out_video),
                "prompt_words": len(str(row.get("prompt") or "").split()),
                "view_type": row.get("view_type", ""),
                "dataset_part": row.get("dataset_part", ""),
                "phase": row.get("phase", ""),
                "notes": note,
            }
        )

    write_metadata(metadata_csv, metadata_rows)
    write_report(report_csv, report_rows)

    print_header("RESULT")
    print_kv("metadata", metadata_csv)
    print_kv("report", report_csv)
    print_kv("ok", ok)
    print_kv("bad", bad)
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
