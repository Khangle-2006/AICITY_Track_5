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
    compute_raft_flow,
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
    parser.add_argument("--limit", type=int, default=None)
    
    # New options
    parser.add_argument("--matcher", default="orb", choices=["orb", "superpoint"])
    parser.add_argument("--depth_model", default="midas", choices=["midas", "depthanything", "depthanything_small", "depthanything_base", "depthanything_large"])
    parser.add_argument("--segment_model", default="deeplabv3", choices=["deeplabv3", "yolov8", "sam2"])
    parser.add_argument("--disocclusion_inpainting", action="store_true")
    parser.add_argument("--alpha_decay", default="none", choices=["none", "linear", "exponential"])
    parser.add_argument("--alpha_decay_rate", type=float, default=0.95)
    parser.add_argument("--alpha_depth_weighting", action="store_true")

    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    cv2_module, _ = import_cv2_np()

    def warmup_directory(path: Path, retries: int = 5):
        import time
        import os
        for i in range(retries):
            try:
                if path.exists():
                    os.listdir(str(path))
                    break
            except Exception:
                time.sleep(0.5)

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_cache.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    device = torch.device("cuda")
    models = PerceptionModels(device, args.model_cache)
    rows = [row for row in read_csv(args.test_csv) if row.get("view_type") == args.view_type]
    if args.limit is not None:
        rows = rows[:args.limit]

    existing_reports = []
    report_path = args.out_dir / "report.csv"
    if not args.clean and report_path.exists():
        try:
            with report_path.open(encoding="utf-8", newline="") as handle:
                existing_reports = list(csv.DictReader(handle))
        except Exception:
            pass

    existing_diagnostics = []
    diag_path = args.out_dir / "diagnostics.csv"
    if not args.clean and diag_path.exists():
        try:
            with diag_path.open(encoding="utf-8", newline="") as handle:
                existing_diagnostics = list(csv.DictReader(handle))
        except Exception:
            pass

    report_rows, diagnostic_rows = [], []
    for position, row in enumerate(rows, start=1):
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        sample_out = args.out_dir / output_name

        if not args.clean and sample_out.exists():
            verified = verify_sample_output(sample_out, target_n)
            if verified.ok:
                match = next((r for r in existing_reports if r.get("output_name") == output_name), None)
                if match is not None:
                    report_rows.append(match)
                else:
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
                diagnostic_rows.extend([d for d in existing_diagnostics if d.get("output_name") == output_name])
                print(f"[{position}/{len(rows)}] {output_name} | already completed & verified, resuming/skipping", flush=True)
                continue

        conditions = load_rgb_paths(cv2_module, numeric_pngs(Path(row["input_dir"])))
        bases = load_rgb_paths(cv2_module, numeric_pngs(args.base_dir / output_name))
        
        if not conditions:
            raise RuntimeError(f"{output_name}: input conditions are missing at {row['input_dir']}")
            
        if len(bases) == 0:
            print(f"[{position}/{len(rows)}] WARNING: {output_name} has 0 base frames. Using duplicated last condition frame as base.")
            last_cond = conditions[-1]
            bases = [last_cond.copy() for _ in range(target_n)]
        
        if len(bases) < target_n:
            print(f"[{position}/{len(rows)}] Padding bases for {output_name} from {len(bases)} to {target_n} by replicating last frame.")
            last_frame = bases[-1]
            while len(bases) < target_n:
                bases.append(last_frame.copy())
        elif len(bases) > target_n:
            bases = bases[:target_n]

        height, width = bases[0].shape[:2]
        anchor = cv2_module.resize(conditions[-1], (width, height), interpolation=cv2_module.INTER_AREA)
        bases = [cv2_module.resize(frame, (width, height), interpolation=cv2_module.INTER_AREA) for frame in bases]
        
        # Custom depth model
        depth = models.depth(anchor, model_type=args.depth_model)
        
        # Custom segmentation model
        masks = models.actor_masks([anchor, *bases], args.segment_batch, model_type=args.segment_model)
        anchor_actor, base_actors = masks[0], masks[1:]
        static_mask = (anchor_actor < 0.1).astype(np.float32)
        intrinsic = camera_matrix(width, height, args.focal_scale)

        raw_poses, pose_diagnostics, raw_qualities = [], [], []
        for base in bases:
            pose, diagnostic = estimate_pose(
                anchor, base, depth, anchor_actor, intrinsic, args.min_matches, args.min_inliers,
                models=models, matcher=args.matcher
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
                
                # Pro-active disocclusion inpainting using RAFT
                if args.disocclusion_inpainting:
                    disocclusion_mask = ((static_mask > 0.5) & (coverage < 0.1)).astype(np.float32)
                    if disocclusion_mask.sum() > 0:
                        flow = compute_raft_flow(models, anchor, base)
                        flow_y, flow_x = flow[..., 1], flow[..., 0]
                        grid_y, grid_x = np.mgrid[0:height, 0:width]
                        map_x = (grid_x + flow_x).astype(np.float32)
                        map_y = (grid_y + flow_y).astype(np.float32)
                        warped = cv2_module.remap(anchor, map_x, map_y, cv2_module.INTER_LINEAR)
                        
                        blend_mask = cv2_module.GaussianBlur(disocclusion_mask, (5, 5), 0.0)[..., None]
                        rendered = (rendered.astype(np.float32) * (1.0 - blend_mask) + warped.astype(np.float32) * blend_mask).astype(np.uint8)
                        coverage = np.clip(coverage + disocclusion_mask, 0.0, 1.0)
                        
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
            
            # Adaptive alpha decay calculation
            effective_alpha = args.alpha
            if args.alpha_decay == "linear":
                effective_alpha = args.alpha * (1.0 - frame_index / len(bases))
            elif args.alpha_decay == "exponential":
                effective_alpha = args.alpha * (args.alpha_decay_rate ** frame_index)
                
            output = compose(
                base, rendered, confidence, effective_alpha,
                depth=depth, use_depth_weighting=args.alpha_depth_weighting
            )
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

    if report_rows:
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
            writer.writeheader()
            writer.writerows(report_rows)
    if diagnostic_rows:
        with diag_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=diagnostic_rows[0].keys())
            writer.writeheader()
            writer.writerows(diagnostic_rows)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
