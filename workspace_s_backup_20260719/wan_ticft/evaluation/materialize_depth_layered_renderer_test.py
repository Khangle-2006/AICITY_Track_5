"""Apply the validated depth-layered renderer to routed test predictions."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output
from wan_ticft.evaluation.materialize_depth_layered_renderer import (
    PerceptionModels,
    camera_matrix,
    compose,
    compose_adaptive,
    estimate_pose,
    motion_mask,
    raw_pose_quality,
    render_confidence,
    smooth_pose_sequence,
    zbuffer_splat,
)


def temporal_alpha(alpha: float, frame_index: int, frame_count: int, enabled: bool, start_frac: float, end_frac: float, floor: float) -> float:
    if not enabled:
        return alpha
    start = int(round(max(0.0, min(1.0, start_frac)) * max(1, frame_count - 1)))
    end = int(round(max(0.0, min(1.0, end_frac)) * max(1, frame_count - 1)))
    if end <= start:
        end = start + 1
    if frame_index <= start:
        return alpha
    progress = min(1.0, (frame_index - start) / float(end - start))
    return alpha + progress * (floor - alpha)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_rgb_paths(cv2_module, paths: list[Path]):
    return [image for path in paths if (image := read_rgb(cv2_module, path)) is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--depth_backend", choices=["dpt", "da3_mono", "da3_metric"], default="dpt")
    parser.add_argument("--da3_model_id", default=None)
    parser.add_argument("--da3_process_res", type=int, default=504)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--adaptive_blend", action="store_true")
    parser.add_argument("--adaptive_motion_sigma", type=float, default=18.0)
    parser.add_argument("--adaptive_min_scale", type=float, default=0.35)
    parser.add_argument("--adaptive_max_weight", type=float, default=0.85)
    parser.add_argument("--temporal_decay", action="store_true")
    parser.add_argument("--temporal_decay_start_frac", type=float, default=0.25)
    parser.add_argument("--temporal_decay_end_frac", type=float, default=1.0)
    parser.add_argument("--temporal_decay_floor", type=float, default=0.35)
    parser.add_argument("--focal_scale", type=float, default=0.9)
    parser.add_argument("--point_stride", type=int, default=2)
    parser.add_argument("--splat_radius", type=int, default=1)
    parser.add_argument("--min_matches", type=int, default=20)
    parser.add_argument("--min_inliers", type=int, default=12)
    parser.add_argument("--photo_sigma", type=float, default=32.0)
    parser.add_argument("--segment_batch", type=int, default=4)
    parser.add_argument("--pose_smoothing_window", type=int, default=7)
    parser.add_argument("--filled_pose_scale", type=float, default=0.35)
    parser.add_argument("--confidence_ema", type=float, default=0.65)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    cv2_module, _ = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_cache.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    device = torch.device("cuda")
    da3_model_id = args.da3_model_id
    if da3_model_id is None:
        da3_model_id = "depth-anything/DA3METRIC-LARGE" if args.depth_backend == "da3_metric" else "depth-anything/DA3MONO-LARGE"
    models = PerceptionModels(
        device,
        args.model_cache,
        depth_backend=args.depth_backend,
        da3_model_id=da3_model_id,
        da3_process_res=args.da3_process_res,
    )
    rows = [row for row in read_csv(args.test_csv) if row.get("view_type") == args.view_type]
    report_rows, diagnostic_rows = [], []
    for position, row in enumerate(rows, start=1):
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        conditions = load_rgb_paths(cv2_module, numeric_pngs(Path(row["input_dir"])))
        bases = load_rgb_paths(cv2_module, numeric_pngs(args.base_dir / output_name))
        if not conditions or len(bases) != target_n:
            raise RuntimeError(
                f"{output_name}: conditions={len(conditions)} base_frames={len(bases)} expected={target_n}"
            )
        height, width = bases[0].shape[:2]
        anchor = cv2_module.resize(conditions[-1], (width, height), interpolation=cv2_module.INTER_AREA)
        bases = [cv2_module.resize(frame, (width, height), interpolation=cv2_module.INTER_AREA) for frame in bases]
        depth = models.depth(anchor)
        masks = models.actor_masks([anchor, *bases], args.segment_batch)
        anchor_actor, base_actors = masks[0], masks[1:]
        static_mask = (anchor_actor < 0.1).astype(np.float32)
        intrinsic = camera_matrix(width, height, args.focal_scale)

        raw_poses, pose_diagnostics, raw_qualities = [], [], []
        for base in bases:
            pose, diagnostic = estimate_pose(
                anchor, base, depth, anchor_actor, intrinsic, args.min_matches, args.min_inliers
            )
            raw_poses.append(pose)
            pose_diagnostics.append(diagnostic)
            raw_qualities.append(raw_pose_quality(diagnostic))
        poses, qualities, filled_flags = smooth_pose_sequence(
            raw_poses, raw_qualities, args.pose_smoothing_window, args.filled_pose_scale
        )

        sample_out = args.out_dir / output_name
        sample_out.mkdir(parents=True, exist_ok=True)
        previous_confidence = None
        previous_base = None
        for frame_index, (base, base_actor, pose, quality, filled) in enumerate(
            zip(bases, base_actors, poses, qualities, filled_flags)
        ):
            diagnostic = pose_diagnostics[frame_index]
            if pose is None or quality <= 0:
                rendered = base
                coverage = np.zeros((height, width), dtype=np.float32)
            else:
                rendered, coverage = zbuffer_splat(
                    anchor, depth, static_mask, pose, intrinsic, args.point_stride, args.splat_radius
                )
            confidence = render_confidence(
                base,
                rendered,
                coverage,
                base_actor,
                quality,
                args.photo_sigma,
                previous_confidence,
                args.confidence_ema,
            )
            effective_alpha = temporal_alpha(
                args.alpha,
                frame_index,
                len(bases),
                args.temporal_decay,
                args.temporal_decay_start_frac,
                args.temporal_decay_end_frac,
                args.temporal_decay_floor,
            )
            motion = motion_mask(base, previous_base, args.adaptive_motion_sigma)
            if args.adaptive_blend:
                output, adaptive_weight = compose_adaptive(
                    base,
                    rendered,
                    confidence,
                    effective_alpha,
                    motion,
                    args.adaptive_min_scale,
                    args.adaptive_max_weight,
                )
            else:
                output = compose(base, rendered, confidence, effective_alpha)
                adaptive_weight = None
            if not write_rgb(cv2_module, sample_out / f"{frame_index}.png", output):
                raise RuntimeError(f"cannot write {sample_out / f'{frame_index}.png'}")
            previous_confidence = confidence
            previous_base = base
            diagnostic_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "output_name": output_name,
                    "frame": frame_index,
                    **diagnostic,
                    "pose_quality": quality,
                    "pose_filled": filled,
                    "coverage": float(coverage.mean()),
                    "actor_fraction": float(base_actor.mean()),
                    "mean_confidence": float(confidence.mean()),
                    "mean_motion": float(motion.mean()),
                    "mean_adaptive_weight": float(adaptive_weight.mean()) if adaptive_weight is not None else "",
                    "effective_alpha": float(effective_alpha),
                }
            )
        verified = verify_sample_output(sample_out, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verification failed: {verified.notes}")
        report_rows.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "view_type": row["view_type"],
                "target_N": target_n,
                "frames_found": verified.frames_found,
                "alpha": f"{args.alpha:.6f}",
                "temporal_decay": args.temporal_decay,
                "temporal_decay_floor": f"{args.temporal_decay_floor:.6f}",
                "status": "ok",
            }
        )
        print(f"[{position}/{len(rows)}] {output_name} | ok", flush=True)

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
