"""Route static-looking BDD clips to repeat-last, otherwise keep depth motion."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from baseline.frame_io import read_rgb
from wan_ticft.evaluation.evaluate_video_reports import read_report


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


def mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    b = resize_like(b, a)
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def history_motion(row: dict) -> float:
    paths = (row.get("frames") or {}).get("condition") or []
    frames = [read_rgb(cv2, Path(path)) for path in paths]
    frames = [frame for frame in frames if frame is not None]
    if len(frames) < 2:
        return 999.0
    return float(np.mean([mean_abs(frames[i], frames[i - 1]) for i in range(1, len(frames))]))


def video_motion(video: Path, frames: int) -> float:
    capture = cv2.VideoCapture(str(video))
    last = None
    diffs = []
    while len(diffs) < frames:
        ok, bgr = capture.read()
        if not ok or bgr is None:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if last is not None:
            diffs.append(mean_abs(rgb, last))
        last = rgb
    capture.release()
    return float(np.mean(diffs)) if diffs else 999.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--depth_report", type=Path, required=True)
    parser.add_argument("--repeat_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--hist_thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--pred_thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--motion_frames", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--copy_videos", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    depth = read_report(args.depth_report)
    repeat = read_report(args.repeat_report)
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == args.view_type and row["sample_id"] in depth]
    if args.limit is not None:
        rows = rows[: args.limit]

    metrics = {}
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        hmot = history_motion(row)
        pmot = video_motion(depth[sample_id], args.motion_frames)
        metrics[sample_id] = {"history_motion": hmot, "prediction_motion": pmot}
        print(f"[{position}/{len(rows)}] {sample_id} history={hmot:.4f} pred={pmot:.4f}", flush=True)

    diagnostic_rows = []
    for hist_threshold in args.hist_thresholds:
        for pred_threshold in args.pred_thresholds:
            label = f"route_h{hist_threshold:g}_p{pred_threshold:g}".replace(".", "p")
            out_dir = args.out_root / label
            video_dir = out_dir / "videos"
            report_rows = []
            repeat_count = 0
            for row in rows:
                sample_id = row["sample_id"]
                item = metrics[sample_id]
                use_repeat = (
                    sample_id in repeat
                    and item["history_motion"] <= hist_threshold
                    and item["prediction_motion"] <= pred_threshold
                )
                source = repeat[sample_id] if use_repeat else depth[sample_id]
                if use_repeat:
                    repeat_count += 1
                if args.copy_videos:
                    video_dir.mkdir(parents=True, exist_ok=True)
                    destination = video_dir / f"{sample_id}{source.suffix}"
                    if not destination.exists():
                        shutil.copy2(source, destination)
                    video = destination
                else:
                    video = source
                report_rows.append(
                    {
                        "sample_id": sample_id,
                        "video": str(video),
                        "status": "ok",
                        "future_frames": "",
                        "route": "repeat" if use_repeat else "depth",
                        **item,
                    }
                )
                diagnostic_rows.append(
                    {
                        "variant": label,
                        "sample_id": sample_id,
                        "route": "repeat" if use_repeat else "depth",
                        **item,
                    }
                )
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
                writer.writeheader()
                writer.writerows(report_rows)
            print(f"{label}: repeat={repeat_count}/{len(rows)} report={out_dir / 'report.csv'}", flush=True)

    if diagnostic_rows:
        with (args.out_root / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=diagnostic_rows[0].keys())
            writer.writeheader()
            writer.writerows(diagnostic_rows)
    print(f"output={args.out_root}")


if __name__ == "__main__":
    main()
