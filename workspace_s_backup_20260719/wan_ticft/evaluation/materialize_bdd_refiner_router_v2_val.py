from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_bdd_ego_motion import (
    combine_predictions,
    load_refiner,
    predict,
    refiner_predictions,
    target_offsets,
)


PHASE_ROUTE = {
    "0": "refiner_blend_25",
    "1": "refiner_blend_50",
    "2": "refiner_blend_25",
    "3": "refiner_blend_50",
    "4": "refiner_blend_25",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2, paths: list[str]):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def source_fps(row: dict) -> float:
    indices = row.get("frame_indices") or {}
    try:
        return float(indices.get("fps") or 30.0)
    except (TypeError, ValueError):
        return 30.0


def history_gap(row: dict) -> int:
    indices = (row.get("frame_indices") or {}).get("condition") or []
    return max(1, int(indices[-1]) - int(indices[-2])) if len(indices) >= 2 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize BDD refiner-router-v2 validation videos.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--estimate_width", type=int, default=640)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    refiner = load_refiner(args.checkpoint)
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"][: args.limit]
    report_rows = []

    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        paths = row.get("frames") or {}
        conditions = load_images(cv2, paths.get("condition") or [])
        targets = load_images(cv2, paths.get("target") or [])
        if len(conditions) < 2 or not targets:
            print(f"[{position}/{len(rows)}] {sample_id} | skip", flush=True)
            continue

        gap = history_gap(row)
        target_indices = target_offsets(row, len(targets))
        full_offsets = list(range(1, max(target_indices) + 1))
        phase = str(row.get("phase", ""))
        route = PHASE_ROUTE.get(phase, "refiner_blend_50")
        classical, _, _ = predict(
            cv2,
            np,
            conditions,
            full_offsets,
            gap,
            {"kind": "phase_router"},
            args.estimate_width,
            phase,
        )
        learned = refiner_predictions(cv2, np, refiner, conditions, full_offsets, gap, phase)
        predictions = combine_predictions(np, classical, learned, full_offsets, route)
        frames = [conditions[-1], *predictions]

        video_path = args.out_dir / "videos" / f"{sample_id}.avi"
        write_video(cv2, frames, video_path, source_fps(row))
        report_rows.append(
            {
                "sample_id": sample_id,
                "video": str(video_path),
                "status": "ok",
                "future_frames": len(predictions),
                "route": route,
                "phase": phase,
            }
        )
        print(f"[{position}/{len(rows)}] {sample_id} | {route} frames={len(frames)}", flush=True)

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "future_frames", "route", "phase"])
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"report={args.out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
