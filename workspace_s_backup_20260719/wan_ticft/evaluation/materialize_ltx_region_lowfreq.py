"""Materialize a low-frequency region correction over an LTX video report."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_bdd_ego_motion import region_predictions
from wan_ticft.evaluation.evaluate_ltx_region_composite_ceiling import read_jsonl
from wan_ticft.evaluation.evaluate_ltx_region_lowfreq import low_frequency_correct


def read_report(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row.get("status") in {"ok", "skipped"} and row.get("video")
        }


def load_conditions(cv2, row):
    paths = (row.get("frames") or {}).get("condition") or []
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=16.0)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--estimate_width", type=int, default=640)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    reports = read_report(args.ltx_report)
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    videos_dir = args.out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    output = []

    for position, row in enumerate(read_jsonl(args.manifest), start=1):
        sample_id = row["sample_id"]
        source = reports.get(sample_id)
        conditions = load_conditions(cv2, row)
        if source is None or len(conditions) < 2:
            print(f"[{position}] {sample_id} | skipped", flush=True)
            continue

        cap = cv2.VideoCapture(source["video"])
        if not cap.isOpened():
            raise RuntimeError(f"{sample_id}: cannot open {source['video']}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or source.get("source_fps") or 30.0)
        ltx = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            ltx.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not ltx:
            raise RuntimeError(f"{sample_id}: no decoded LTX frames")

        indices = (row.get("frame_indices") or {}).get("condition") or []
        history_gap = max(1, int(indices[-1]) - int(indices[-2])) if len(indices) >= 2 else 1
        source_fps = float((row.get("frame_indices") or {}).get("fps") or fps)
        offsets = [max(1, int(round((index + 1) * source_fps / fps))) for index in range(len(ltx))]
        region = region_predictions(
            cv2, np, conditions[-2], conditions[-1], offsets, history_gap, args.estimate_width
        )
        region = [
            frame
            if frame.shape[:2] == ltx_frame.shape[:2]
            else cv2.resize(
                frame,
                (ltx_frame.shape[1], ltx_frame.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            for frame, ltx_frame in zip(region, ltx)
        ]
        corrected = [
            low_frequency_correct(cv2, np, ltx_frame, region_frame, args.sigma, args.alpha)
            for ltx_frame, region_frame in zip(ltx, region)
        ]
        video_path = videos_dir / f"{sample_id}.mp4"
        write_video(cv2, corrected, video_path, fps)
        output.append(
            {
                "sample_id": sample_id,
                "video": str(video_path),
                "status": "ok",
                "source_fps": source.get("source_fps", ""),
                "future_frames": len(corrected),
                "sigma": args.sigma,
                "alpha": args.alpha,
            }
        )
        print(f"[{position}] {sample_id} | ok", flush=True)

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(f"rows={len(output)}")
    print(f"report={args.out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
