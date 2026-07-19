from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def load_ati_track_module(ati_dir: Path):
    path = ati_dir / "tools" / "get_track_from_videos.py"
    spec = importlib.util.spec_from_file_location("ati_get_track_from_videos", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def extrapolate_tracks(
    observed: np.ndarray,
    width: int,
    height: int,
    future_frames: int,
    velocity_window: int,
    velocity_decay: float,
) -> tuple[np.ndarray, dict[str, float]]:
    # observed: [T, N, 3] containing x, y, visibility.
    xy = observed[..., :2].astype(np.float32)
    visible = observed[..., 2] > 0.5
    last_xy = xy[-1]
    start = max(1, len(xy) - velocity_window)
    deltas = xy[start:] - xy[start - 1 : -1]
    valid_pairs = visible[start:] & visible[start - 1 : -1]

    velocity = np.zeros_like(last_xy)
    for point in range(last_xy.shape[0]):
        valid = deltas[:, point][valid_pairs[:, point]]
        if len(valid):
            velocity[point] = np.median(valid, axis=0)

    tracks = np.zeros((last_xy.shape[0], future_frames, 1, 3), dtype=np.float32)
    position = last_xy.copy()
    active = visible[-1].copy()
    current_velocity = velocity.copy()

    for frame in range(future_frames):
        if frame:
            position = position + current_velocity
            current_velocity *= velocity_decay
        in_bounds = (
            (position[:, 0] >= 0)
            & (position[:, 0] < width)
            & (position[:, 1] >= 0)
            & (position[:, 1] < height)
        )
        active &= in_bounds
        tracks[:, frame, 0, :2] = np.clip(position, [0, 0], [width - 1, height - 1])
        tracks[:, frame, 0, 2] = active.astype(np.float32)

    speed = np.linalg.norm(velocity, axis=1)
    displacement = np.linalg.norm(tracks[:, -1, 0, :2] - tracks[:, 0, 0, :2], axis=1)
    stats = {
        "median_observed_velocity_px": float(np.median(speed)),
        "p90_observed_velocity_px": float(np.percentile(speed, 90)),
        "median_future_displacement_px": float(np.median(displacement)),
        "p90_future_displacement_px": float(np.percentile(displacement, 90)),
        "final_visible_fraction": float(tracks[:, -1, 0, 2].mean()),
    }
    return tracks, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ati_dir", type=Path, default=Path("/workspace/third_party/ATI"))
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_points", type=int, default=40)
    parser.add_argument("--future_frames", type=int, default=121)
    parser.add_argument("--velocity_window", type=int, default=8)
    parser.add_argument("--velocity_decay", type=float, default=0.99)
    args = parser.parse_args()

    ati = load_ati_track_module(args.ati_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = args.out_dir / "tracks"
    images_dir = args.out_dir / "images"
    tracks_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    metadata = {}
    with (args.input_dir / "report.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            metadata[row["sample"]] = row

    tracker = ati.TrackAnyPoint(n_points=args.num_points)
    records = []
    report = []

    videos = sorted((args.input_dir / "history_videos").glob("*.mp4"))
    if not videos:
        raise RuntimeError("no history videos found")

    for index, video_path in enumerate(videos):
        sample = video_path.stem
        print(f"[{index + 1}/{len(videos)}] {sample}", flush=True)
        frames = ati.load_video_to_frames(str(video_path), preset_fps=15)
        if len(frames) < 2:
            raise RuntimeError(f"{sample}: fewer than two frames")

        observed = tracker(frames)[0]
        width, height = frames[-1].size
        future, stats = extrapolate_tracks(
            observed,
            width,
            height,
            args.future_frames,
            args.velocity_window,
            args.velocity_decay,
        )

        image_path = images_dir / f"{sample}.png"
        track_path = tracks_dir / f"{sample}.pth"
        frames[-1].save(image_path)
        ati.array_to_npz_bytes(future, track_path)

        row = metadata.get(sample, {})
        prompt = " ".join(
            part.strip()
            for part in (row.get("caption_pedestrian", ""), row.get("caption_vehicle", ""))
            if part.strip()
        )
        records.append({"image": str(image_path), "text": prompt, "track": str(track_path)})
        report.append(
            {
                "sample": sample,
                "history_frames": len(frames),
                "width": width,
                "height": height,
                **stats,
            }
        )

    with (args.out_dir / "test.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(records, f, sort_keys=False, allow_unicode=True)
    with (args.out_dir / "track_report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(report),
                "future_frames": args.future_frames,
                "num_points": args.num_points,
                "velocity_window": args.velocity_window,
                "velocity_decay": args.velocity_decay,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
