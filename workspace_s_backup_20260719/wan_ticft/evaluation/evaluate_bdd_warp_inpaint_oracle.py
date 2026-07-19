"""Measure the ceiling of future-flow warp plus LTX hole filling on BDD."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.generators import compute_flow, warp_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2, paths: list[str]):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def resize_like(cv2, image, target):
    if image.shape[:2] == target.shape[:2]:
        return image
    return cv2.resize(image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)


def forward_splat(np, image, flow):
    """Bilinear forward splat with average collision handling."""
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    dst_x = grid_x + flow[:, :, 0]
    dst_y = grid_y + flow[:, :, 1]
    x0 = np.floor(dst_x).astype(np.int32)
    y0 = np.floor(dst_y).astype(np.int32)
    dx = dst_x - x0
    dy = dst_y - y0

    accum = np.zeros((height * width, 3), dtype=np.float32)
    weights = np.zeros(height * width, dtype=np.float32)
    source = image.reshape(-1, 3).astype(np.float32)
    for x, y, weight in (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x0 + 1, y0, dx * (1.0 - dy)),
        (x0, y0 + 1, (1.0 - dx) * dy),
        (x0 + 1, y0 + 1, dx * dy),
    ):
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (weight > 1e-6)
        indices = (y[valid] * width + x[valid]).reshape(-1)
        values = weight[valid].reshape(-1)
        np.add.at(weights, indices, values)
        for channel in range(3):
            np.add.at(accum[:, channel], indices, source[valid.reshape(-1), channel] * values)

    visible = weights.reshape(height, width) > 1e-6
    output = np.zeros((height * width, 3), dtype=np.float32)
    output[weights > 1e-6] = accum[weights > 1e-6] / weights[weights > 1e-6, None]
    return np.clip(output.reshape(height, width, 3), 0, 255).astype(np.uint8), visible


def dilate_holes(cv2, np, visible, radius: int):
    holes = (~visible).astype(np.uint8)
    if radius > 0:
        size = 2 * radius + 1
        holes = cv2.dilate(holes, np.ones((size, size), dtype=np.uint8))
    return holes.astype(bool)


def composite(np, warp, fill, holes):
    return np.where(holes[:, :, None], fill, warp)


def score(cv2, np, predictions, targets):
    metrics = [
        (
            psnr(cv2, np, prediction, target),
            ssim_gray(cv2, np, prediction, target),
            mean_absdiff(np, prediction, target),
        )
        for prediction, target in zip(predictions, targets)
    ]
    return tuple(float(np.mean([value[index] for value in metrics])) for index in range(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--flow_width", type=int, default=640)
    parser.add_argument("--hole_dilate", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    videos = read_report(args.ltx_report)
    rows_out = []
    processed = 0
    for position, row in enumerate(read_jsonl(args.manifest), start=1):
        if row.get("view_type") != "bdd":
            continue
        if args.limit and processed >= args.limit:
            break
        sample_id = row["sample_id"]
        video = videos.get(sample_id)
        paths = row.get("frames") or {}
        conditions = load_images(cv2, paths.get("condition") or [])
        targets = load_images(cv2, paths.get("target") or [])
        if video is None or not conditions or not targets:
            print(f"[{position}] {sample_id} | skipped", flush=True)
            continue

        generated_fps = video_fps(cv2, video)
        mapped = map_target_offsets(row, generated_fps)
        frame_map = read_video_rgb_frames(cv2, video, mapped)
        ltx_frames = [frame_map[index] for index in mapped if index in frame_map]
        if len(ltx_frames) != len(targets):
            print(f"[{position}] {sample_id} | incomplete LTX", flush=True)
            continue

        last = conditions[-1]
        candidates = {
            "repeat_last": [],
            "ltx16": [],
            "oracle_backward_flow": [],
            "oracle_forward_splat_black": [],
            **{f"oracle_splat_repeat_hole_d{radius}": [] for radius in args.hole_dilate},
            **{f"oracle_splat_ltx_hole_d{radius}": [] for radius in args.hole_dilate},
            **{f"oracle_splat_target_hole_d{radius}": [] for radius in args.hole_dilate},
        }
        hole_fractions = {radius: [] for radius in args.hole_dilate}
        for target, ltx in zip(targets, ltx_frames):
            last_frame = resize_like(cv2, last, target)
            ltx_frame = resize_like(cv2, ltx, target)
            forward = compute_flow(cv2, np, last_frame, target, args.flow_width)
            splat, visible = forward_splat(np, last_frame, forward)
            backward = compute_flow(cv2, np, target, last_frame, args.flow_width)
            backward_prediction = warp_rgb(cv2, np, last_frame, backward, -1.0)

            candidates["repeat_last"].append(last_frame)
            candidates["ltx16"].append(ltx_frame)
            candidates["oracle_backward_flow"].append(backward_prediction)
            candidates["oracle_forward_splat_black"].append(splat)
            for radius in args.hole_dilate:
                holes = dilate_holes(cv2, np, visible, radius)
                hole_fractions[radius].append(float(np.mean(holes)))
                candidates[f"oracle_splat_repeat_hole_d{radius}"].append(
                    composite(np, splat, last_frame, holes)
                )
                candidates[f"oracle_splat_ltx_hole_d{radius}"].append(
                    composite(np, splat, ltx_frame, holes)
                )
                candidates[f"oracle_splat_target_hole_d{radius}"].append(
                    composite(np, splat, target, holes)
                )

        for candidate, predictions in candidates.items():
            p, s, d = score(cv2, np, predictions, targets)
            radius = next(
                (value for value in args.hole_dilate if candidate.endswith(f"_d{value}")),
                None,
            )
            rows_out.append(
                {
                    "sample_id": sample_id,
                    "phase": row.get("phase", ""),
                    "candidate": candidate,
                    "target_count": len(targets),
                    "hole_fraction": (
                        f"{np.mean(hole_fractions[radius]):.6f}" if radius is not None else ""
                    ),
                    "psnr": f"{p:.6f}",
                    "ssim": f"{s:.6f}",
                    "absdiff": f"{d:.6f}",
                }
            )
        processed += 1
        print(f"[{processed}] {sample_id} | ok", flush=True)

    if not rows_out:
        raise RuntimeError("no BDD samples evaluated")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)

    candidates = list(dict.fromkeys(row["candidate"] for row in rows_out))
    print(f"{'Candidate':<34} {'N':>5} {'Hole%':>8} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 83)
    for candidate in candidates:
        items = [row for row in rows_out if row["candidate"] == candidate]
        holes = [float(row["hole_fraction"]) for row in items if row["hole_fraction"]]
        print(
            f"{candidate:<34} {len(items):>5} "
            f"{100.0 * np.mean(holes) if holes else 0.0:>8.2f} "
            f"{np.mean([float(row['psnr']) for row in items]):>10.4f} "
            f"{np.mean([float(row['ssim']) for row in items]):>10.4f} "
            f"{np.mean([float(row['absdiff']) for row in items]):>10.4f}"
        )
    print(f"output {args.out_csv}")


if __name__ == "__main__":
    main()
