"""Forward-splat optical-flow extrapolation gate for BDD.

Unlike backward remapping, forward splatting moves last-frame pixels to their
predicted future locations and exposes disocclusion holes explicitly. Holes are
filled with the real last frame. This smoke tests whether splatting reduces the
rubbery warp artifacts seen in naive flow extrapolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.generators import compute_flow
from wan_ticft.evaluation.evaluate_video_reports import source_fps, target_offsets
from wan_ticft.evaluation.materialize_depth_layered_renderer import write_video


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2, paths: list[str]):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def resize_like(cv2, image, reference):
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


def parse_variant(value: str) -> tuple[str, float, float, int]:
    parts = value.split(":")
    if len(parts) != 4:
        raise ValueError("variant must be label:decay:max_scale:hole_dilate")
    label, raw_decay, raw_max_scale, raw_hole_dilate = parts
    return label, float(raw_decay), float(raw_max_scale), int(raw_hole_dilate)


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


def frame_scales(row: dict, count: int) -> list[float]:
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    if len(condition) >= 2:
        dt = max(1.0, float(int(condition[-1]) - int(condition[-2])))
        return [float(index + 1) / dt for index in range(count)]
    return [float(index + 1) for index in range(count)]


def forward_splat(cv2, np, image, flow, scale: float):
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
        src_index = np.flatnonzero(valid.reshape(-1))
        dst_index = (y[valid] * width + x[valid]).reshape(-1)
        value = wgt[valid].reshape(-1).astype(np.float32)
        np.add.at(weights, dst_index, value)
        for channel in range(3):
            np.add.at(accum[:, channel], dst_index, flat[src_index, channel] * value)
    visible = weights > 1e-6
    out = np.zeros_like(accum)
    out[visible] = accum[visible] / weights[visible, None]
    return out.reshape(height, width, 3).clip(0, 255).astype(np.uint8), visible.reshape(height, width)


def fill_holes(cv2, np, splat, visible, fill, dilate: int):
    holes = (~visible).astype(np.uint8)
    if dilate > 0:
        holes = cv2.dilate(holes, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8), iterations=1)
    mask = holes.astype(bool)
    return np.where(mask[..., None], fill, splat).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--flow_width", type=int, default=640)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument(
        "--variant",
        action="append",
        default=[
            "splat_linear2_d0:1.0:2.0:0",
            "splat_linear2_d2:1.0:2.0:2",
            "splat_decay070_d0:0.70:3.0:0",
            "splat_decay070_d2:0.70:3.0:2",
        ],
    )
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == args.view_type][: args.limit]
    variants = [parse_variant(value) for value in args.variant]
    reports = {label: [] for label, _, _, _ in variants}
    with (args.out_root / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for position, row in enumerate(rows, start=1):
        sid = row["sample_id"]
        conditions = load_images(cv2, (row.get("frames") or {}).get("condition") or [])
        offsets = target_offsets(row)
        count = max(offsets) if offsets else int((row.get("ticft") or {}).get("target_frames") or 9)
        if len(conditions) < 2 or count <= 0:
            print(f"[{position}/{len(rows)}] {sid} | skip", flush=True)
            continue
        previous = conditions[-2]
        last = resize_like(cv2, conditions[-1], previous)
        flow = compute_flow(cv2, np, previous, last, args.flow_width)
        raw_scales = frame_scales(row, count)
        fps = source_fps(row) or 30.0
        for label, decay, max_scale, hole_dilate in variants:
            frames = []
            for raw_scale in raw_scales:
                splat, visible = forward_splat(cv2, np, last, flow, decay_scale(raw_scale, decay, max_scale))
                frames.append(fill_holes(cv2, np, splat, visible, last, hole_dilate))
            video = args.out_root / label / "videos" / f"{sid}.avi"
            write_video(cv2, frames, video, fps, prepend_guard=True)
            reports[label].append({"sample_id": sid, "video": str(video), "status": "ok", "future_frames": len(frames)})
        print(f"[{position}/{len(rows)}] {sid} | ok", flush=True)

    for label, rows_out in reports.items():
        path = args.out_root / label / "report.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "future_frames"])
            writer.writeheader()
            writer.writerows(rows_out)
    print(f"output={args.out_root}")


if __name__ == "__main__":
    main()
