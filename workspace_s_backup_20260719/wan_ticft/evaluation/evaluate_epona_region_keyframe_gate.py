#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.generators import compute_flow, make_region_motion_mask
from wan_ticft.evaluation.evaluate_manifest_motion_baselines import (
    CONFIGS,
    load_consecutive_source_targets,
    predict,
    score_sequence,
)


SAMPLE_TO_SLIDING = {
    "val_bdd_pc_5k_video1407_phase0": 0,
    "val_bdd_pc_5k_video3237_phase0": 1,
    "val_bdd_pc_5k_video5447_phase3": 2,
    "val_bdd_pc_5k_video7040_phase0": 3,
}


def read_manifest(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["sample_id"]: row for row in rows}


def blend_keyframe(
    np,
    cv2,
    region,
    keyframe,
    anchor_offset,
    decay_frames,
    residual_mask=None,
):
    keyframe = cv2.resize(
        keyframe,
        (region[anchor_offset - 1].shape[1], region[anchor_offset - 1].shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    residual = keyframe.astype(np.float32) - region[anchor_offset - 1].astype(np.float32)
    if residual_mask is not None:
        residual = residual * residual_mask[:, :, None]
    output = []
    for offset, base in enumerate(region, start=1):
        if offset <= anchor_offset:
            weight = offset / float(anchor_offset)
        else:
            weight = max(0.0, 1.0 - (offset - anchor_offset) / float(decay_frames))
        pred = base.astype(np.float32) + weight * residual
        output.append(np.clip(pred, 0, 255).astype(np.uint8))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--epona_root", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--future_frames", type=int, default=24)
    parser.add_argument("--anchor_offset", type=int, default=6)
    parser.add_argument("--decay_frames", type=int, default=12)
    parser.add_argument("--motion_masked", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    manifest = read_manifest(args.manifest)
    motion_args = SimpleNamespace(
        flow_width=426,
        global_estimate_width=640,
        region_motion_tau=1.5,
        region_motion_temp=0.75,
        region_max_alpha=0.85,
        region_overhead_max_alpha=0.45,
        region_mask_blur=3.0,
        condition_indices=[],
    )
    rows = []
    for sample_id, sliding_index in SAMPLE_TO_SLIDING.items():
        row = manifest[sample_id]
        paths = row["frames"]
        conditions = [read_rgb(cv2, Path(path)) for path in paths["condition"]]
        conditions = [frame for frame in conditions if frame is not None]
        height, width = conditions[-1].shape[:2]
        targets = load_consecutive_source_targets(
            cv2, row, args.future_frames, height, width
        )
        indices = row["frame_indices"]["condition"]
        history_gap = max(1, int(indices[-1]) - int(indices[-2]))
        motion_args.condition_indices = indices
        offsets = list(range(1, args.future_frames + 1))
        region = predict(
            cv2,
            np,
            conditions,
            offsets,
            history_gap,
            "bdd",
            CONFIGS["region_norm_medium"],
            motion_args,
        )
        residual_mask = None
        if args.motion_masked:
            flow = compute_flow(
                cv2, np, conditions[-2], conditions[-1], motion_args.flow_width
            )
            residual_mask, _ = make_region_motion_mask(
                cv2, np, flow, "bdd", motion_args
            )
        keyframe = read_rgb(
            cv2, args.epona_root / f"sliding_{sliding_index}" / "10.png"
        )
        if keyframe is None or len(targets) != args.future_frames:
            raise RuntimeError(f"{sample_id}: missing keyframe or targets")
        hybrid = blend_keyframe(
            np,
            cv2,
            region,
            keyframe,
            args.anchor_offset,
            args.decay_frames,
            residual_mask,
        )
        region_metrics = score_sequence(cv2, np, region, targets)
        hybrid_metrics = score_sequence(cv2, np, hybrid, targets)
        rows.append(
            {
                "sample_id": sample_id,
                "region_psnr": region_metrics[0],
                "hybrid_psnr": hybrid_metrics[0],
                "delta_psnr": hybrid_metrics[0] - region_metrics[0],
                "region_ssim": region_metrics[1],
                "hybrid_ssim": hybrid_metrics[1],
                "delta_ssim": hybrid_metrics[1] - region_metrics[1],
                "region_absdiff": region_metrics[2],
                "hybrid_absdiff": hybrid_metrics[2],
                "motion_masked": args.motion_masked,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "samples": len(rows),
        "wins": sum(row["delta_psnr"] > 0 for row in rows),
        "region_psnr": float(np.mean([row["region_psnr"] for row in rows])),
        "hybrid_psnr": float(np.mean([row["hybrid_psnr"] for row in rows])),
        "delta_psnr": float(np.mean([row["delta_psnr"] for row in rows])),
        "delta_ssim": float(np.mean([row["delta_ssim"] for row in rows])),
        "worst_delta_psnr": float(min(row["delta_psnr"] for row in rows)),
    }
    print(json.dumps(summary, indent=2))
    for row in rows:
        print(
            f"{row['sample_id']}: dPSNR={row['delta_psnr']:+.4f} "
            f"dSSIM={row['delta_ssim']:+.4f}"
        )


if __name__ == "__main__":
    main()
