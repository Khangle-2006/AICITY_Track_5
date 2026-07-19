"""Apply motion-persistence routing to BDD test folders."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from baseline.frame_io import numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


def mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    b = resize_like(b, a)
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def history_motion(input_dir: Path) -> float:
    frames = [read_rgb(cv2, path) for path in numeric_pngs(input_dir)]
    frames = [frame for frame in frames if frame is not None]
    if len(frames) < 2:
        return 999.0
    return float(np.mean([mean_abs(frames[i], frames[i - 1]) for i in range(1, len(frames))]))


def prediction_motion(sample_dir: Path, frames: int) -> float:
    paths = numeric_pngs(sample_dir)[: max(2, frames)]
    images = [read_rgb(cv2, path) for path in paths]
    images = [image for image in images if image is not None]
    if len(images) < 2:
        return 999.0
    return float(np.mean([mean_abs(images[i], images[i - 1]) for i in range(1, len(images))]))


def write_repeat(anchor: np.ndarray, out_dir: Path, target_n: int, reference_shape: tuple[int, int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = reference_shape
    frame = cv2.resize(anchor, (width, height), interpolation=cv2.INTER_AREA)
    for index in range(target_n):
        if not write_rgb(cv2, out_dir / f"{index}.png", frame):
            raise RuntimeError(f"cannot write {out_dir / f'{index}.png'}")


def copy_sample(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--depth_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--hist_threshold", type=float, default=4.0)
    parser.add_argument("--pred_threshold", type=float, default=1.0)
    parser.add_argument("--motion_frames", type=int, default=12)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    rows = [row for row in read_csv(args.test_csv) if row.get("view_type") == args.view_type]
    report_rows, diagnostic_rows = [], []
    for position, row in enumerate(rows, start=1):
        output_name = row["source_folder"]
        sample_id = row["sample_id"]
        target_n = int(row["target_N"])
        depth_sample = args.depth_dir / output_name
        out_sample = args.out_dir / output_name
        depth_frames = numeric_pngs(depth_sample)
        if len(depth_frames) != target_n:
            raise RuntimeError(f"{output_name}: depth_frames={len(depth_frames)} expected={target_n}")
        conditions = [read_rgb(cv2, path) for path in numeric_pngs(Path(row["input_dir"]))]
        conditions = [frame for frame in conditions if frame is not None]
        if not conditions:
            raise RuntimeError(f"{output_name}: no readable history frames")
        reference = read_rgb(cv2, depth_frames[0])
        if reference is None:
            raise RuntimeError(f"{output_name}: unreadable depth frame {depth_frames[0]}")
        hmot = history_motion(Path(row["input_dir"]))
        pmot = prediction_motion(depth_sample, args.motion_frames)
        route = "repeat" if hmot <= args.hist_threshold and pmot <= args.pred_threshold else "depth"
        if route == "repeat":
            write_repeat(conditions[-1], out_sample, target_n, reference.shape[:2])
        else:
            copy_sample(depth_sample, out_sample)
        verified = verify_sample_output(out_sample, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verification failed: {verified.notes}")
        report_rows.append(
            {
                "output_name": output_name,
                "sample_id": sample_id,
                "view_type": row["view_type"],
                "target_N": target_n,
                "frames_found": verified.frames_found,
                "route": route,
                "history_motion": f"{hmot:.6f}",
                "prediction_motion": f"{pmot:.6f}",
                "status": "ok",
            }
        )
        diagnostic_rows.append(report_rows[-1])
        print(f"[{position}/{len(rows)}] {output_name} | {route} history={hmot:.4f} pred={pmot:.4f}", flush=True)

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)
    with (args.out_dir / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostic_rows[0].keys())
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
