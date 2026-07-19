from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_report(path: Path) -> dict[str, Path]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: Path(row["video"]) for row in csv.DictReader(handle) if row.get("status", "ok") == "ok"}


def read_video(path: Path, max_frames: int = 90):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return frames, fps if fps > 1 else 30.0


def resize(frame, width: int, height: int):
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def label(frame, text: str):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (360, 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_report", type=Path, required=True)
    parser.add_argument("--medium_report", type=Path, required=True)
    parser.add_argument("--candidate_report", type=Path, required=True)
    parser.add_argument("--sample_id", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    target = read_report(args.target_report)
    medium = read_report(args.medium_report)
    cand = read_report(args.candidate_report)
    out_frames = []
    fps = 10.0
    for sid in args.sample_id:
        videos = []
        for report in (target, medium, cand):
            frames, fps = read_video(report[sid])
            videos.append(frames)
        count = min(len(v) for v in videos)
        if count == 0:
            continue
        h, w = videos[0][0].shape[:2]
        for idx in range(count):
            cells = [
                label(resize(videos[0][idx], w, h), f"{sid} target"),
                label(resize(videos[1][idx], w, h), "medium baseline"),
                label(resize(videos[2][idx], w, h), "masked refiner"),
            ]
            row = np.concatenate(cells, axis=1)
            out_frames.append(row)
        spacer = np.zeros_like(out_frames[-1])
        out_frames.extend([spacer] * 5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    h, w = out_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {args.out}")
    for frame in out_frames:
        writer.write(frame)
    writer.release()
    print(args.out)


if __name__ == "__main__":
    main()
