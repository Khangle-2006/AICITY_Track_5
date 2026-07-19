from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_report(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row.get("sample_id") or row.get("source_folder"): row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok"}


def numeric_pngs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)


def load_source(row: dict[str, str], max_frames: int = 90) -> list[np.ndarray]:
    raw = row.get("video") or row.get("source_dir") or ""
    path = Path(raw)
    frames = []
    if path.is_dir():
        for item in numeric_pngs(path)[:max_frames]:
            frame = cv2.imread(str(item), cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        return frames
    cap = cv2.VideoCapture(str(path))
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (420, 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_report", type=Path, required=True)
    parser.add_argument("--right_report", type=Path, required=True)
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--left_label", default="left")
    parser.add_argument("--right_label", default="right")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    left_rows = read_report(args.left_report)
    right_rows = read_report(args.right_report)
    left_row = left_rows.get(args.sample_id)
    right_row = right_rows.get(args.sample_id)
    if left_row is None or right_row is None:
        raise RuntimeError(f"missing sample {args.sample_id}")
    left = load_source(left_row)
    right = load_source(right_row)
    count = min(len(left), len(right))
    if count == 0:
        raise RuntimeError("empty input")
    h, w = left[0].shape[:2]
    frames = []
    for idx in range(count):
        r = right[idx]
        if r.shape[:2] != (h, w):
            r = cv2.resize(r, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(np.concatenate([label(left[idx], args.left_label), label(r, args.right_label)], axis=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (frames[0].shape[1], frames[0].shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {args.out}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(args.out)


if __name__ == "__main__":
    main()
