from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np


def read_report(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("status") == "ok"}


def read_video(cv2, path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames, fps


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def parse_variant(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise ValueError(f"variant must be label=alpha_b, got {value!r}")
    label, raw = value.split("=", 1)
    alpha_b = float(raw)
    if not 0.0 <= alpha_b <= 1.0:
        raise ValueError(f"alpha_b out of range: {value!r}")
    return label, alpha_b


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix two video reports into new report variants.")
    parser.add_argument("--report_a", type=Path, required=True, help="Base report, e.g. CNN.")
    parser.add_argument("--report_b", type=Path, required=True, help="Mix target report, e.g. LTX.")
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="label=alpha_b")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    report_a = read_report(args.report_a)
    report_b = read_report(args.report_b)
    variants = [parse_variant(value) for value in args.variant]

    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    common = sorted(set(report_a) & set(report_b))
    if not common:
        raise RuntimeError("reports have no common sample_id")

    for label, alpha_b in variants:
        rows = []
        for index, sample_id in enumerate(common, start=1):
            frames_a, fps = read_video(cv2, Path(report_a[sample_id]["video"]))
            frames_b, fps_b = read_video(cv2, Path(report_b[sample_id]["video"]))
            fps = fps or fps_b
            count = min(len(frames_a), len(frames_b))
            mixed = []
            for frame_a, frame_b in zip(frames_a[:count], frames_b[:count]):
                if frame_a.shape[:2] != frame_b.shape[:2]:
                    frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]), interpolation=cv2.INTER_CUBIC)
                mixed.append(
                    np.clip(
                        (1.0 - alpha_b) * frame_a.astype(np.float32) + alpha_b * frame_b.astype(np.float32),
                        0,
                        255,
                    ).astype(np.uint8)
                )
            path = args.out_root / label / "videos" / f"{sample_id}.mp4"
            write_video(cv2, mixed, path, fps)
            rows.append(
                {
                    "sample_id": sample_id,
                    "video": str(path),
                    "status": "ok",
                    "source_fps": f"{fps:.6f}",
                    "future_frames": len(mixed),
                    "mix_alpha_b": f"{alpha_b:.6f}",
                }
            )
            print(f"{label} [{index}/{len(common)}] {sample_id} | ok", flush=True)
        report_path = args.out_root / label / "report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
