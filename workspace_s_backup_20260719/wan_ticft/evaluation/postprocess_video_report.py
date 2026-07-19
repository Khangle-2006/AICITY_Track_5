"""Apply deterministic camera-style postprocessing to videos in a report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baseline.frame_io import import_cv2_np
from wan_ticft.evaluation.compute_real_fid import degrade


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["sample_id", "video", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process_video(cv2, np, src: Path, dst: Path, blur: float, noise: float, jpeg_q: int, downscale: float, seed: int) -> int:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"FFV1"),
        fps,
        (width, height),
    )
    rng = np.random.default_rng(seed)
    count = 0
    deg = blur > 0 or noise > 0 or jpeg_q < 100 or downscale < 1.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if deg:
            frame = degrade(frame, blur, noise, jpeg_q, downscale, rng)
        writer.write(frame)
        count += 1
    cap.release()
    writer.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--out_report", type=Path, required=True)
    parser.add_argument("--blur", type=float, default=0.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--jpeg_q", type=int, default=100)
    parser.add_argument("--downscale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows_out = []
    for index, row in enumerate(read_rows(args.report), start=1):
        video = Path(row.get("video", ""))
        sample_id = row.get("sample_id") or video.stem
        out_row = dict(row)
        if row.get("status") != "ok" or not video.exists():
            out_row["status"] = "bad"
            rows_out.append(out_row)
            print(f"[{index}] {sample_id} skipped", flush=True)
            continue
        dst = args.out_dir / f"{sample_id}.avi"
        frame_count = process_video(
            cv2,
            np,
            video,
            dst,
            args.blur,
            args.noise,
            args.jpeg_q,
            args.downscale,
            args.seed + index * 1009,
        )
        out_row["video"] = str(dst)
        out_row["status"] = "ok" if frame_count else "bad"
        if "future_frames" in out_row:
            out_row["future_frames"] = str(frame_count)
        elif "frames" in out_row:
            out_row["frames"] = str(frame_count)
        rows_out.append(out_row)
        print(f"[{index}] {sample_id} frames={frame_count}", flush=True)

    write_rows(args.out_report, rows_out)
    print(f"report={args.out_report}")


if __name__ == "__main__":
    main()
