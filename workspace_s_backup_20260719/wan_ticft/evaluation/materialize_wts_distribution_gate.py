"""Materialize bounded WTS target and motion-candidate videos for distribution audit."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np
from baseline.generators import compute_flow
from wan_ticft.evaluation.evaluate_manifest_motion_baselines import (
    CONFIGS,
    load_consecutive_source_targets,
    load_images,
    predict,
    read_jsonl,
)


DEFAULT_CANDIDATES = ("repeat_last", "region_norm_medium", "region_norm_long12", "region_norm_long24")
VIEWS = ("bdd", "wts_overhead_or_fixed_camera", "wts_vehicle_or_ip_camera")


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in [frames[0], *frames]:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def actor_layer_small_stats(cv2, np, conditions, args) -> dict[str, float]:
    last = conditions[-1]
    prev = conditions[-2] if len(conditions) >= 2 else conditions[-1]
    background = np.median(np.stack(conditions, axis=0).astype(np.float32), axis=0)
    flow = compute_flow(cv2, np, prev, last, args.flow_width)
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    color_delta = np.mean(np.abs(last.astype(np.float32) - background), axis=2)

    motion_tau = 0.20
    color_tau = 8.0
    color_high = 24.0
    min_area = 20.0
    max_area_frac = 0.025

    seed = ((mag > motion_tau) & (color_delta > color_tau)) | (color_delta > color_high)
    mask = seed.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    image_area = float(h * w)
    max_area = max_area_frac * image_area
    kept_areas = []
    raw_areas = []
    filled = np.zeros_like(mask)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        raw_areas.append(area)
        if min_area <= area <= max_area:
            kept_areas.append(area)
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

    source_mask = cv2.GaussianBlur(filled.astype(np.float32) / 255.0, (0, 0), sigmaX=1.0)
    source_mask = np.clip(source_mask, 0.0, 1.0)
    kept_sum = float(sum(kept_areas))
    raw_sum = float(sum(raw_areas))
    return {
        "actor_mask_mean": float(np.mean(source_mask)),
        "actor_mask_p90": float(np.percentile(source_mask, 90)),
        "actor_mask_sum_frac": kept_sum / image_area,
        "actor_mask_max_frac": (max(kept_areas) / image_area) if kept_areas else 0.0,
        "actor_component_count": float(len(kept_areas)),
        "actor_raw_sum_frac": raw_sum / image_area,
        "actor_raw_component_count": float(len(raw_areas)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit_per_domain", type=int, default=10)
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument(
        "--cover_manifest_targets",
        action="store_true",
        help="Use the minimum per-sample length that covers its farthest target offset.",
    )
    parser.add_argument("--candidates", nargs="+", choices=tuple(CONFIGS), default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--flow_width", type=int, default=426)
    parser.add_argument("--global_estimate_width", type=int, default=640)
    parser.add_argument("--region_motion_tau", type=float, default=1.5)
    parser.add_argument("--region_motion_temp", type=float, default=0.75)
    parser.add_argument("--region_max_alpha", type=float, default=0.85)
    parser.add_argument("--region_overhead_max_alpha", type=float, default=0.45)
    parser.add_argument("--region_mask_blur", type=float, default=3.0)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts = {view: 0 for view in args.views}
    reports = {name: [] for name in ("target", *args.candidates)}
    skipped = []
    selected_rows = []
    scanned = 0
    total_required = args.limit_per_domain * len(args.views)

    for row in read_jsonl(args.manifest):
        view = row.get("view_type")
        if view not in counts or counts[view] >= args.limit_per_domain:
            continue
        scanned += 1
        paths = row.get("frames") or {}
        conditions = load_images(cv2, paths.get("condition") or [])
        if len(conditions) < 2:
            skipped.append((row["sample_id"], view, f"conditions={len(conditions)}"))
            print(f"[skip] {row['sample_id']} | conditions={len(conditions)}", flush=True)
            continue
        height, width = conditions[-1].shape[:2]
        frame_count = args.frames
        if args.cover_manifest_targets:
            frame_indices = row.get("frame_indices") or {}
            condition_indices = frame_indices.get("condition") or []
            target_indices = frame_indices.get("target") or []
            if condition_indices and target_indices:
                last_condition = int(condition_indices[-1])
                # Some repaired manifests contain repeated timestamp targets where
                # the target index equals the last condition index. The evaluator
                # maps those to generated frame 1, so materialize at least one
                # future frame instead of passing an empty offset list downstream.
                frame_count = max(1, max(int(index) - last_condition for index in target_indices))
        targets = load_consecutive_source_targets(cv2, row, frame_count, height, width)
        if len(targets) != frame_count:
            skipped.append((row["sample_id"], view, f"targets={len(targets)} expected={frame_count}"))
            print(
                f"[skip] {row['sample_id']} | targets={len(targets)} expected={frame_count}",
                flush=True,
            )
            continue

        indices = (row.get("frame_indices") or {}).get("condition") or []
        history_gap = max(1, int(indices[-1]) - int(indices[-2])) if len(indices) >= 2 else 1
        args.condition_indices = indices
        offsets = list(range(1, frame_count + 1))
        fps = float((row.get("frame_indices") or {}).get("fps") or 30.0)
        frames_by_name = {"target": targets}
        stats_by_name = {"target": {}}
        for candidate in args.candidates:
            frames_by_name[candidate] = predict(
                cv2,
                np,
                conditions,
                offsets,
                history_gap,
                row["view_type"],
                CONFIGS[candidate],
                args,
            )
            stats_by_name[candidate] = (
                actor_layer_small_stats(cv2, np, conditions, args)
                if candidate == "actor_layer_small_filled"
                else {}
            )

        for name, frames in frames_by_name.items():
            path = args.out_dir / name / f"{row['sample_id']}.mp4"
            write_video(cv2, frames, path, fps)
            stats = stats_by_name.get(name, {})
            reports[name].append(
                {
                    "sample_id": row["sample_id"],
                    "view_type": row["view_type"],
                    "video": str(path),
                    "status": "ok",
                    "frames": len(frames),
                    "fps": fps,
                    **{key: f"{value:.8f}" for key, value in stats.items()},
                }
            )
        counts[view] += 1
        selected_rows.append(row)
        position = sum(counts.values())
        print(f"[{position}/{total_required}] {row['sample_id']} | ok", flush=True)

        if all(count >= args.limit_per_domain for count in counts.values()):
            break

    missing = {
        view: args.limit_per_domain - count for view, count in counts.items() if count < args.limit_per_domain
    }
    if missing:
        raise RuntimeError(
            f"manifest does not contain enough valid WTS rows: {missing}; "
            f"scanned={scanned} skipped={len(skipped)}"
        )

    for name, rows in reports.items():
        path = args.out_dir / f"{name}_report.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            fields = []
            for row in rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    (args.out_dir / "selected_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected_rows),
        encoding="utf-8",
    )
    if skipped:
        skip_path = args.out_dir / "skipped_rows.csv"
        with skip_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "view_type", "reason"])
            writer.writerows(skipped)
        print(f"skipped={len(skipped)}")
        print(f"skipped_rows={skip_path}")
    print(f"rows={sum(counts.values())}")
    print(f"counts={counts}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
