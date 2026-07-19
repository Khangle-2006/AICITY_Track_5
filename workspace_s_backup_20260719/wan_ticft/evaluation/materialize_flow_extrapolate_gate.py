"""Materialize BDD future frames by extrapolating optical flow from conditions.

This is a blur diagnostic / VFI-principle gate: predictions are single-source
warps from the last real condition frame, avoiding alpha averaging across
multiple candidate images. It tests whether the main blur is caused by
averaging, and whether simple future flow is accurate enough to be useful.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.generators import compute_flow, warp_rgb
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


def parse_variant(value: str) -> tuple[str, float, float]:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("variant must be label:decay:max_scale")
    label, raw_decay, raw_max_scale = parts
    return label, float(raw_decay), float(raw_max_scale)


def scaled_offsets(row: dict, count: int) -> list[float]:
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    target = indices.get("target") or []
    if len(condition) >= 2:
        dt = max(1.0, float(int(condition[-1]) - int(condition[-2])))
        last = int(condition[-1])
        if target and count == len(target):
            return [max(0.0, (float(int(index) - last) / dt)) for index in target[:count]]
        return [max(0.0, float(index + 1) / dt) for index in range(count)]
    return [float(index + 1) for index in range(count)]


def decay_scale(raw_scale: float, decay: float, max_scale: float) -> float:
    if decay >= 0.999:
        return min(raw_scale, max_scale)
    # Sum a geometric motion sequence for non-integer horizons.
    whole = int(raw_scale)
    frac = raw_scale - whole
    total = 0.0
    step = 1.0
    for _ in range(whole):
        total += step
        step *= decay
    total += frac * step
    return min(total, max_scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--flow_width", type=int, default=640)
    parser.add_argument(
        "--variant",
        action="append",
        default=[
            "linear_cap1:1.0:1.0",
            "linear_cap2:1.0:2.0",
            "linear_cap4:1.0:4.0",
            "decay085_cap4:0.85:4.0",
            "decay070_cap3:0.70:3.0",
        ],
        help="label:decay:max_scale",
    )
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    if args.clean and args.out_root.exists():
        import shutil

        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == args.view_type][: args.limit]
    variants = [parse_variant(value) for value in args.variant]
    reports: dict[str, list[dict[str, object]]] = {label: [] for label, _, _ in variants}

    selected_manifest = args.out_root / "selected_manifest.jsonl"
    with selected_manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        frame_paths = row.get("frames") or {}
        conditions = load_images(cv2, frame_paths.get("condition") or [])
        offsets = target_offsets(row)
        target_count = max(offsets) if offsets else int((row.get("ticft") or {}).get("target_frames") or 9)
        if len(conditions) < 2 or target_count <= 0:
            print(f"[{position}/{len(rows)}] {sample_id} | skip", flush=True)
            continue
        previous = conditions[-2]
        last = resize_like(cv2, conditions[-1], previous)
        flow = compute_flow(cv2, np, previous, last, args.flow_width)
        scales = scaled_offsets(row, target_count)
        fps = source_fps(row) or 30.0

        for label, decay, max_scale in variants:
            frames = [warp_rgb(cv2, np, last, flow, decay_scale(scale, decay, max_scale)) for scale in scales]
            out_path = args.out_root / label / "videos" / f"{sample_id}.avi"
            write_video(cv2, frames, out_path, fps, prepend_guard=True)
            reports[label].append(
                {
                    "sample_id": sample_id,
                    "video": str(out_path),
                    "status": "ok",
                    "future_frames": len(frames),
                    "decay": decay,
                    "max_scale": max_scale,
                }
            )
        print(f"[{position}/{len(rows)}] {sample_id} | ok", flush=True)

    for label, rows_out in reports.items():
        report_path = args.out_root / label / "report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["sample_id", "video", "status", "future_frames", "decay", "max_scale"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)

    print(f"output={args.out_root}")


if __name__ == "__main__":
    main()
