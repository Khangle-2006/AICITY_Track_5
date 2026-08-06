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
    estimate_pose,
    raw_pose_quality,
    render_confidence,
    smooth_pose_sequence,
    zbuffer_splat,
)


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
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--alpha", type=float, default=0.75)
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
    models = PerceptionModels(device, args.model_cache)
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
            output = compose(base, rendered, confidence, args.alpha)
            if not write_rgb(cv2_module, sample_out / f"{frame_index}.png", output):
                raise RuntimeError(f"cannot write {sample_out / f'{frame_index}.png'}")
            previous_confidence = confidence
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
