from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .console import kv, section
from .frame_io import import_cv2_np, numeric_pngs


def is_frame_dir(path: Path) -> bool:
    return path.is_dir() and bool(numeric_pngs(path))


def discover_frame_dirs(root: Path) -> list[Path]:
    if is_frame_dir(root):
        return [root]
    out = []
    for path in sorted(root.iterdir()):
        if path.name.startswith("_"):
            continue
        if is_frame_dir(path):
            out.append(path)
    return out


def read_first_frame(cv2, frames: list[Path]):
    img = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read first frame: {frames[0]}")
    return img


def make_writer(cv2, out_path: Path, fps: float, width: int, height: int, codec: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter: {out_path} codec={codec}")
    return writer


def frames_to_video(cv2, frame_dir: Path, out_path: Path, fps: float, codec: str, resize_width: int = 0, resize_height: int = 0) -> dict[str, Any]:
    frames = numeric_pngs(frame_dir)
    if not frames:
        return {"frame_dir": str(frame_dir), "out_path": str(out_path), "status": "failed", "frames": 0, "notes": "no numeric png frames"}

    first = read_first_frame(cv2, frames)
    h, w = first.shape[:2]
    if resize_width > 0 and resize_height > 0:
        w, h = resize_width, resize_height

    writer = make_writer(cv2, out_path, fps, w, h, codec)
    written = 0
    failed = 0
    for path in frames:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            failed += 1
            continue
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(img)
        written += 1
    writer.release()

    return {
        "frame_dir": str(frame_dir),
        "out_path": str(out_path),
        "status": "ok" if written == len(frames) else "partial",
        "frames": written,
        "failed": failed,
        "width": w,
        "height": h,
        "fps": fps,
        "notes": "",
    }


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["frame_dir", "out_path", "status", "frames", "failed", "width", "height", "fps", "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="A frame folder or a root containing sample frame folders.")
    parser.add_argument("--out", required=True, help="Output mp4 path for one folder, or output directory for many folders.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--codec", default="mp4v", help="Common options: mp4v, avc1, XVID")
    parser.add_argument("--resize_width", type=int, default=0)
    parser.add_argument("--resize_height", type=int, default=0)
    parser.add_argument("--report_csv", default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    frames_root = Path(args.frames).expanduser()
    out = Path(args.out).expanduser()
    frame_dirs = discover_frame_dirs(frames_root)
    if not frame_dirs:
        raise SystemExit(f"No frame folders found under: {frames_root}")

    section("FRAMES TO VIDEO")
    kv("frames", frames_root)
    kv("out", out)
    kv("frame dirs", len(frame_dirs))
    kv("fps", args.fps)
    kv("codec", args.codec)

    rows = []
    many = len(frame_dirs) > 1 or out.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}
    for frame_dir in frame_dirs:
        if many:
            out_path = out / f"{frame_dir.name}.mp4"
        else:
            out_path = out
        rows.append(
            frames_to_video(
                cv2,
                frame_dir,
                out_path,
                fps=args.fps,
                codec=args.codec,
                resize_width=args.resize_width,
                resize_height=args.resize_height,
            )
        )

    report_csv = Path(args.report_csv).expanduser() if args.report_csv else (out if many else out.parent) / "frames_to_video_report.csv"
    write_report(report_csv, rows)

    section("RESULT")
    kv("videos written", sum(1 for r in rows if r["status"] in {"ok", "partial"}))
    kv("failures", sum(1 for r in rows if r["status"] == "failed"))
    kv("report_csv", report_csv)
    if len(rows) <= 10:
        for row in rows:
            print(f"  - {row['status']} | {row['frames']} frames | {row['out_path']}")


if __name__ == "__main__":
    main()
