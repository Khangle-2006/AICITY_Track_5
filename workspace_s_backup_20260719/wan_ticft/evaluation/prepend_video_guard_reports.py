"""Prepend a duplicate guard frame to generated-video reports without rerunning inference."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baseline.frame_io import import_cv2_np


def parse_report(raw: str) -> tuple[str, Path]:
    label, path = raw.split("=", 1)
    return label, Path(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_all(cv2, path: Path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


def write_guarded(cv2, frames, fps: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in [frames[0], *frames]:
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, metavar="LABEL=CSV")
    parser.add_argument("--out_root", type=Path, required=True)
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    args.out_root.mkdir(parents=True, exist_ok=True)
    for raw in args.report:
        label, report_path = parse_report(raw)
        output_rows = []
        rows = read_rows(report_path)
        for position, row in enumerate(rows, start=1):
            source = Path(row["video"])
            frames, fps = read_all(cv2, source)
            if not frames:
                raise RuntimeError(f"cannot decode {source}")
            destination = args.out_root / label / "videos" / f"{row['sample_id']}.avi"
            write_guarded(cv2, frames, fps, destination)
            updated = dict(row)
            updated["video"] = str(destination)
            if "future_frames" in updated:
                updated["future_frames"] = str(len(frames))
            output_rows.append(updated)
            print(f"{label} [{position}/{len(rows)}] {row['sample_id']} | ok", flush=True)
        output_report = args.out_root / label / "report.csv"
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with output_report.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)
    print(f"output={args.out_root}")


if __name__ == "__main__":
    main()
