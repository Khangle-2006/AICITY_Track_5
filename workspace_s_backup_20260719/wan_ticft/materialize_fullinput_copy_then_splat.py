"""Build a test candidate that copies known input frames, then flow-splats future frames.

The previous full-input splat branch extrapolated from frame 0 onward, which warps
known input frames and makes the whole sequence visually invalid. This variant
keeps every available input frame exactly as-is and only predicts frames after
the input clip ends.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from baseline.frame_io import numeric_pngs, read_rgb, write_rgb
from baseline.generators import compute_flow
from baseline.verify import verify_sample_output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def decay_scale(raw_scale: float, decay: float, max_scale: float) -> float:
    if decay >= 0.999:
        return min(raw_scale, max_scale)
    whole = int(raw_scale)
    frac = raw_scale - whole
    total = 0.0
    step = 1.0
    for _ in range(whole):
        total += step
        step *= decay
    total += frac * step
    return min(total, max_scale)


def forward_splat(image: np.ndarray, flow: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    dst_x = grid_x + flow[..., 0] * scale
    dst_y = grid_y + flow[..., 1] * scale
    x0 = np.floor(dst_x).astype(np.int32)
    y0 = np.floor(dst_y).astype(np.int32)
    dx = dst_x - x0
    dy = dst_y - y0
    flat = image.reshape(-1, 3).astype(np.float32)
    accum = np.zeros((height * width, 3), dtype=np.float32)
    weights = np.zeros(height * width, dtype=np.float32)
    for x, y, wgt in (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x0 + 1, y0, dx * (1.0 - dy)),
        (x0, y0 + 1, (1.0 - dx) * dy),
        (x0 + 1, y0 + 1, dx * dy),
    ):
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (wgt > 1e-6)
        source_index = np.flatnonzero(valid.reshape(-1))
        target_index = (y[valid] * width + x[valid]).reshape(-1)
        value = wgt[valid].reshape(-1).astype(np.float32)
        np.add.at(weights, target_index, value)
        for channel in range(3):
            np.add.at(accum[:, channel], target_index, flat[source_index, channel] * value)
    visible = weights > 1e-6
    out = np.zeros_like(accum)
    out[visible] = accum[visible] / weights[visible, None]
    return out.reshape(height, width, 3).clip(0, 255).astype(np.uint8), visible.reshape(height, width)


def fill_holes(splat: np.ndarray, visible: np.ndarray, fill: np.ndarray, dilate: int) -> np.ndarray:
    holes = (~visible).astype(np.uint8)
    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        holes = cv2.dilate(holes, kernel, iterations=1)
    return np.where(holes.astype(bool)[..., None], fill, splat).astype(np.uint8)


def resize_to_720p(frame: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == (720, 1280):
        return frame
    return cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)


def build_manifest_lookup(manifest: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for row in read_jsonl(manifest):
        lookup[row["sample_id"]] = row
        lookup[row["source_folder"]] = row
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--flow_width", type=int, default=640)
    parser.add_argument("--decay", type=float, default=0.70)
    parser.add_argument("--max_scale", type=float, default=3.0)
    parser.add_argument("--hole_dilate", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2.setNumThreads(4)
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_lookup = build_manifest_lookup(args.manifest)
    rows = read_csv(args.report)
    out_report: list[dict[str, str | int]] = []

    for position, row in enumerate(rows, start=1):
        output_name = row["output_name"]
        target_n = int(row["target_N"])
        manifest_row = manifest_lookup.get(row["sample_id"]) or manifest_lookup.get(row["source_folder"])
        if manifest_row is None:
            raise RuntimeError(f"{output_name}: missing manifest row")

        sample_out = args.out_dir / output_name
        existing = verify_sample_output(sample_out, target_n)
        if existing.ok and not args.clean:
            out_report.append(
                {
                    "output_name": output_name,
                    "sample_id": row["sample_id"],
                    "source_folder": row["source_folder"],
                    "view_type": row["view_type"],
                    "target_N": target_n,
                    "input_frames": "skip",
                    "future_frames": "skip",
                    "frames_found": existing.frames_found,
                    "status": "ok",
                    "notes": "skip_existing",
                }
            )
            print(f"[{position}/{len(rows)}] {output_name} | skip(existing)", flush=True)
            continue

        input_paths = numeric_pngs(Path(manifest_row["input_dir"]))
        frames = [resize_to_720p(read_rgb(cv2, path)) for path in input_paths]
        frames = [frame for frame in frames if frame is not None]
        if len(frames) < 2:
            raise RuntimeError(f"{output_name}: need at least two input frames, got {len(frames)}")

        previous, last = frames[-2], frames[-1]
        flow = compute_flow(cv2, np, previous, last, args.flow_width)
        sample_out.mkdir(parents=True, exist_ok=True)

        copied = min(len(frames), target_n)
        future = max(0, target_n - copied)
        for index in range(copied):
            write_rgb(cv2, sample_out / f"{index}.png", frames[index])
        for future_index in range(future):
            scale = decay_scale(float(future_index + 1), args.decay, args.max_scale)
            splat, visible = forward_splat(last, flow, scale)
            pred = fill_holes(splat, visible, last, args.hole_dilate)
            write_rgb(cv2, sample_out / f"{copied + future_index}.png", pred)

        verify = verify_sample_output(sample_out, target_n)
        if not verify.ok:
            raise RuntimeError(f"{output_name}: verify failed {verify.notes}")
        out_report.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "source_folder": row["source_folder"],
                "view_type": row["view_type"],
                "target_N": target_n,
                "input_frames": len(frames),
                "future_frames": future,
                "frames_found": verify.frames_found,
                "status": "ok",
                "notes": "",
            }
        )
        print(
            "[{}/{}] {} input={} target={} future={} | ok".format(
                position, len(rows), output_name, len(frames), target_n, future
            ),
            flush=True,
        )

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "output_name",
            "sample_id",
            "source_folder",
            "view_type",
            "target_N",
            "input_frames",
            "future_frames",
            "frames_found",
            "status",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_report)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
