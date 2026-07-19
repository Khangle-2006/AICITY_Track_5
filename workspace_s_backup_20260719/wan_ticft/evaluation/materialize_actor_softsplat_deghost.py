"""Actor deghosting with forward splatting over an existing video report.

This is the occlusion-aware counterpart to materialize_actor_flow_deghost.py:
instead of backward-sampling anchor actors into the generated frame, it pushes
anchor pixels forward with dense flow and composites only validated dynamic
actor components. The goal is to remove duplicated moving cars/pedestrians
without globally filtering the scaffold.
"""
from __future__ import annotations

import argparse
import csv
import os as _os
import shutil
from pathlib import Path

_T = _os.environ.get("CPU_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, _T)

import cv2
import numpy as np
import torch

import cv2 as _cv2
from baseline.frame_io import import_cv2_np
from wan_ticft.evaluation.evaluate_video_reports import read_report, video_fps
from wan_ticft.evaluation.materialize_actor_flow_deghost import (
    clean_actor_alpha,
    color_match_paste_region,
    component_validated_alpha,
    dense_flow_base_to_anchor,
    dynamic_anchor_mask,
    erase_nearby_base_ghosts,
)
from wan_ticft.evaluation.materialize_depth_layered_renderer import (
    PerceptionModels,
    load_full_video_frames,
    load_images,
    read_jsonl,
    write_video,
)

_cv2.setNumThreads(int(_T))


def forward_splat_weighted(npm, values: np.ndarray, flow: np.ndarray, source_weight: np.ndarray, metric: np.ndarray):
    """Forward splat arbitrary channels using bilinear weights and softmax metric."""
    height, width = source_weight.shape[:2]
    channels = values.shape[2]
    grid_x, grid_y = npm.meshgrid(
        npm.arange(width, dtype=npm.float32),
        npm.arange(height, dtype=npm.float32),
    )
    dst_x = grid_x + flow[..., 0]
    dst_y = grid_y + flow[..., 1]
    x0 = npm.floor(dst_x).astype(npm.int32)
    y0 = npm.floor(dst_y).astype(npm.int32)
    dx = dst_x - x0
    dy = dst_y - y0

    flat_values = values.reshape(-1, channels).astype(npm.float32)
    flat_src = source_weight.reshape(-1).astype(npm.float32)
    flat_metric = metric.reshape(-1).astype(npm.float32)
    accum = npm.zeros((height * width, channels), dtype=npm.float32)
    weights = npm.zeros(height * width, dtype=npm.float32)

    for x, y, bilinear in (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x0 + 1, y0, dx * (1.0 - dy)),
        (x0, y0 + 1, (1.0 - dx) * dy),
        (x0 + 1, y0 + 1, dx * dy),
    ):
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (bilinear > 1e-6) & (source_weight > 1e-4)
        if not npm.any(valid):
            continue
        src_index = npm.flatnonzero(valid.reshape(-1))
        dst_index = (y[valid] * width + x[valid]).reshape(-1)
        metric_valid = npm.clip(flat_metric[src_index], -30.0, 30.0)
        weight = bilinear[valid].reshape(-1).astype(npm.float32) * flat_src[src_index] * npm.exp(metric_valid)
        npm.add.at(weights, dst_index, weight)
        for channel in range(channels):
            npm.add.at(accum[:, channel], dst_index, flat_values[src_index, channel] * weight)

    out = npm.zeros_like(accum)
    visible = weights > 1e-6
    out[visible] = accum[visible] / weights[visible, None]
    return out.reshape(height, width, channels), weights.reshape(height, width)


def splat_anchor_actor(cv2m, npm, anchor: np.ndarray, anchor_mask: np.ndarray, base: np.ndarray, flow_scale: float, flow_penalty: float):
    flow_ab = dense_flow_base_to_anchor(cv2m, anchor, base, flow_scale)
    flow_mag = npm.sqrt(flow_ab[..., 0] ** 2 + flow_ab[..., 1] ** 2)
    source_weight = npm.clip(anchor_mask, 0.0, 1.0).astype(npm.float32)
    metric = 2.0 * source_weight - flow_penalty * flow_mag
    splat, weights = forward_splat_weighted(npm, anchor.astype(npm.float32), flow_ab, source_weight, metric)
    mask_values = source_weight[..., None].astype(npm.float32)
    splat_mask, _ = forward_splat_weighted(npm, mask_values, flow_ab, source_weight, metric)
    visible = (weights > 0.04).astype(npm.float32)
    return npm.clip(splat, 0, 255).astype(npm.uint8), npm.clip(splat_mask[..., 0], 0.0, 1.0), visible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--flow_scale", type=float, default=0.5)
    parser.add_argument("--flow_penalty", type=float, default=0.015)
    parser.add_argument("--actor_strength", type=float, default=0.90)
    parser.add_argument("--mask_dilate", type=int, default=15)
    parser.add_argument("--mask_blur", type=float, default=4.0)
    parser.add_argument("--segment_batch", type=int, default=4)
    parser.add_argument("--tag", default="actor_softsplat")
    parser.add_argument("--dynamic_only", action="store_true")
    parser.add_argument("--dynamic_residual_thresh", type=float, default=2.5)
    parser.add_argument("--dynamic_min_area", type=int, default=80)
    parser.add_argument("--dynamic_max_area_frac", type=float, default=0.02)
    parser.add_argument("--dynamic_max_bbox_height_frac", type=float, default=0.22)
    parser.add_argument("--dynamic_exclude_bottom_frac", type=float, default=0.03)
    parser.add_argument("--component_validate", action="store_true")
    parser.add_argument("--component_min_area", type=int, default=70)
    parser.add_argument("--component_max_area_frac", type=float, default=0.035)
    parser.add_argument("--component_max_bbox_height_frac", type=float, default=0.30)
    parser.add_argument("--component_exclude_bottom_frac", type=float, default=0.02)
    parser.add_argument("--component_boundary", type=int, default=15)
    parser.add_argument("--component_max_boundary_diff", type=float, default=30.0)
    parser.add_argument("--component_min_valid_frac", type=float, default=0.80)
    parser.add_argument("--component_color_match", action="store_true")
    parser.add_argument("--color_match_boundary", type=int, default=19)
    parser.add_argument("--color_match_max_shift", type=float, default=18.0)
    parser.add_argument("--erase_base_ghosts", action="store_true")
    parser.add_argument("--erase_mask_thresh", type=float, default=0.35)
    parser.add_argument("--erase_alpha_keep_thresh", type=float, default=0.06)
    parser.add_argument("--erase_near_radius", type=int, default=95)
    parser.add_argument("--erase_min_area", type=int, default=70)
    parser.add_argument("--erase_max_area_frac", type=float, default=0.02)
    parser.add_argument("--erase_inpaint_radius", type=float, default=3.0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2m, _ = import_cv2_np()
    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    out_dir = args.out_root / args.tag
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_jsonl(args.manifest) if r.get("view_type") == args.view_type][: args.limit]
    base_report = read_report(args.base_report)
    models = PerceptionModels(torch.device("cuda"), args.model_cache, depth_backend="dpt")
    report = []

    for pos, row in enumerate(rows, 1):
        sid = row["sample_id"]
        base_video = base_report.get(sid)
        conditions = load_images(cv2m, (row.get("frames") or {}).get("condition", []))
        if not base_video or len(conditions) < 1:
            print(f"[{pos}/{len(rows)}] {sid} | skip", flush=True)
            continue
        bases = load_full_video_frames(cv2m, base_video)
        if not bases:
            print(f"[{pos}/{len(rows)}] {sid} | skip(no base)", flush=True)
            continue

        h, w = bases[0].shape[:2]
        anchor = cv2m.resize(conditions[-1], (w, h), interpolation=cv2m.INTER_AREA)
        extra_history = [cv2m.resize(conditions[-2], (w, h), interpolation=cv2m.INTER_AREA)] if len(conditions) >= 2 else []
        masks = models.actor_masks([*extra_history, anchor, *bases], args.segment_batch)
        if extra_history:
            previous_mask, anchor_mask, base_masks = masks[0], masks[1], masks[2:]
            if args.dynamic_only:
                anchor_mask = dynamic_anchor_mask(
                    cv2m,
                    extra_history[0],
                    anchor,
                    previous_mask,
                    anchor_mask,
                    args.flow_scale,
                    args.dynamic_residual_thresh,
                    args.dynamic_min_area,
                    args.dynamic_max_area_frac,
                    args.dynamic_max_bbox_height_frac,
                    args.dynamic_exclude_bottom_frac,
                )
        else:
            anchor_mask, base_masks = masks[0], masks[1:]

        outputs = []
        accepted_components = 0
        total_components = 0
        alpha_means = []
        erase_means = []
        for base, base_mask in zip(bases, base_masks):
            splat_actor, splat_mask, splat_valid = splat_anchor_actor(
                cv2m,
                np,
                anchor,
                anchor_mask,
                base,
                args.flow_scale,
                args.flow_penalty,
            )
            if args.component_validate:
                alpha, diagnostics = component_validated_alpha(
                    cv2m,
                    np,
                    base,
                    splat_actor,
                    splat_mask,
                    splat_valid,
                    args.component_min_area,
                    args.component_max_area_frac,
                    args.component_max_bbox_height_frac,
                    args.component_exclude_bottom_frac,
                    args.component_boundary,
                    args.component_max_boundary_diff,
                    args.component_min_valid_frac,
                    args.mask_dilate,
                    args.mask_blur,
                )
                total_components += len(diagnostics)
                accepted_components += sum(1 for item in diagnostics if item["accepted"] > 0.5)
            else:
                alpha = clean_actor_alpha(cv2m, np.zeros_like(base_mask), splat_mask, args.mask_dilate, args.mask_blur)

            base_eff = base
            erase_fraction = 0.0
            if args.erase_base_ghosts:
                base_eff, erase_fraction = erase_nearby_base_ghosts(
                    cv2m,
                    np,
                    base,
                    base_mask,
                    alpha,
                    args.erase_mask_thresh,
                    args.erase_alpha_keep_thresh,
                    args.erase_near_radius,
                    args.erase_min_area,
                    args.erase_max_area_frac,
                    args.erase_inpaint_radius,
                )
            if args.component_color_match:
                splat_actor = color_match_paste_region(
                    cv2m,
                    np,
                    base_eff,
                    splat_actor,
                    alpha,
                    args.color_match_boundary,
                    args.color_match_max_shift,
                )
            alpha_means.append(float(np.mean(alpha)))
            erase_means.append(erase_fraction)
            weight = (alpha * args.actor_strength)[..., None]
            out = np.clip(base_eff.astype(np.float32) * (1.0 - weight) + splat_actor.astype(np.float32) * weight, 0, 255)
            outputs.append(out.astype(np.uint8))

        fps = video_fps(cv2m, base_video) or 30.0
        vp = out_dir / "videos" / f"{sid}.avi"
        write_video(cv2m, outputs, vp, fps, prepend_guard=False)
        report.append(
            {
                "sample_id": sid,
                "video": str(vp),
                "status": "ok",
                "future_frames": len(outputs),
                "accepted_components": accepted_components,
                "total_components": total_components,
                "alpha_mean": f"{float(np.mean(alpha_means)):.6f}" if alpha_means else "0",
                "erase_mean": f"{float(np.mean(erase_means)):.6f}" if erase_means else "0",
            }
        )
        print(
            f"[{pos}/{len(rows)}] {sid} | ok (frames={len(outputs)} "
            f"components={accepted_components}/{total_components} "
            f"alpha={float(np.mean(alpha_means)) if alpha_means else 0:.4f} "
            f"erase={float(np.mean(erase_means)) if erase_means else 0:.4f})",
            flush=True,
        )

    rp = out_dir / "report.csv"
    with rp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "video",
                "status",
                "future_frames",
                "accepted_components",
                "total_components",
                "alpha_mean",
                "erase_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(report)
    print(f"output={out_dir}")


if __name__ == "__main__":
    main()
