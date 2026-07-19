"""Actor-flow deghost pass over an existing BDD video report.

This is a cheap gate candidate: keep the current method's background and
replace only actor-like regions with real anchor pixels warped by dense optical
flow. It targets the dominant failure of cnn_ltxmix/multiframe outputs: moving
actors appearing twice after blends disagree.
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
from wan_ticft.evaluation.materialize_depth_layered_renderer import (
    PerceptionModels,
    load_full_video_frames,
    load_images,
    read_jsonl,
    write_video,
)

_cv2.setNumThreads(int(_T))


def dense_flow_base_to_anchor(cv2m, base: np.ndarray, anchor: np.ndarray, scale: float) -> np.ndarray:
    """Return flow on base-grid pointing to corresponding anchor coordinates."""
    h, w = base.shape[:2]
    if scale != 1.0:
        sw, sh = max(16, int(w * scale)), max(16, int(h * scale))
        base_s = cv2m.resize(base, (sw, sh), interpolation=cv2m.INTER_AREA)
        anchor_s = cv2m.resize(anchor, (sw, sh), interpolation=cv2m.INTER_AREA)
    else:
        sw, sh = w, h
        base_s, anchor_s = base, anchor
    g0 = cv2m.cvtColor(base_s, cv2m.COLOR_RGB2GRAY)
    g1 = cv2m.cvtColor(anchor_s, cv2m.COLOR_RGB2GRAY)
    flow = cv2m.calcOpticalFlowFarneback(
        g0,
        g1,
        None,
        0.5,
        4,
        31,
        4,
        7,
        1.5,
        0,
    )
    if (sw, sh) != (w, h):
        flow = cv2m.resize(flow, (w, h), interpolation=cv2m.INTER_LINEAR)
        flow[..., 0] *= w / sw
        flow[..., 1] *= h / sh
    return flow.astype(np.float32)


def warp_from_anchor(cv2m, anchor: np.ndarray, flow_ba: np.ndarray, border_value=0):
    h, w = anchor.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx + flow_ba[..., 0]
    map_y = yy + flow_ba[..., 1]
    return cv2m.remap(
        anchor,
        map_x,
        map_y,
        interpolation=cv2m.INTER_LINEAR,
        borderMode=cv2m.BORDER_CONSTANT,
        borderValue=border_value,
    )


def warp_valid_mask(cv2m, shape: tuple[int, int], flow_ba: np.ndarray) -> np.ndarray:
    valid = np.ones(shape, dtype=np.float32)
    warped = warp_from_anchor(cv2m, valid, flow_ba, border_value=0)
    return np.clip(warped, 0.0, 1.0)


def estimate_global_affine(cv2m, src: np.ndarray, dst: np.ndarray, src_actor: np.ndarray, min_matches: int = 20):
    """Estimate src->dst affine using non-actor ORB matches."""
    g0 = cv2m.cvtColor(src, cv2m.COLOR_RGB2GRAY)
    g1 = cv2m.cvtColor(dst, cv2m.COLOR_RGB2GRAY)
    orb = cv2m.ORB_create(nfeatures=2500, fastThreshold=8)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None:
        return None
    pairs = cv2m.BFMatcher(cv2m.NORM_HAMMING).knnMatch(d0, d1, k=2)
    src_pts, dst_pts = [], []
    h, w = src_actor.shape[:2]
    for pair in pairs:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance >= 0.72 * second.distance:
            continue
        x, y = k0[first.queryIdx].pt
        xi, yi = int(round(x)), int(round(y))
        if src_actor[min(max(yi, 0), h - 1), min(max(xi, 0), w - 1)] > 0.1:
            continue
        src_pts.append(k0[first.queryIdx].pt)
        dst_pts.append(k1[first.trainIdx].pt)
    if len(src_pts) < min_matches:
        return None
    affine, _ = cv2m.estimateAffinePartial2D(
        np.float32(src_pts),
        np.float32(dst_pts),
        method=cv2m.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=2000,
        confidence=0.995,
    )
    return affine


def dynamic_anchor_mask(
    cv2m,
    previous: np.ndarray,
    anchor: np.ndarray,
    previous_actor: np.ndarray,
    anchor_actor: np.ndarray,
    flow_scale: float,
    residual_thresh: float,
    min_area: int,
    max_area_frac: float,
    max_bbox_height_frac: float,
    exclude_bottom_frac: float,
) -> np.ndarray:
    """Actor pixels whose history flow is not explained by global camera motion."""
    h, w = anchor_actor.shape[:2]
    affine = estimate_global_affine(cv2m, anchor, previous, anchor_actor)
    if affine is None:
        return (anchor_actor > 0.25).astype(np.float32)
    flow_ap = dense_flow_base_to_anchor(cv2m, anchor, previous, flow_scale)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    global_x = affine[0, 0] * xx + affine[0, 1] * yy + affine[0, 2]
    global_y = affine[1, 0] * xx + affine[1, 1] * yy + affine[1, 2]
    actual_x = xx + flow_ap[..., 0]
    actual_y = yy + flow_ap[..., 1]
    residual = np.sqrt((actual_x - global_x) ** 2 + (actual_y - global_y) ** 2)
    raw = ((anchor_actor > 0.22) | (previous_actor > 0.35)) & (residual > residual_thresh)
    raw = raw.astype(np.uint8)
    count, labels, stats, _ = cv2m.connectedComponentsWithStats(raw, connectivity=8)
    keep = np.zeros_like(raw)
    max_area = int(max_area_frac * h * w) if max_area_frac > 0 else h * w
    max_bbox_height = int(max_bbox_height_frac * h) if max_bbox_height_frac > 0 else h
    bottom_cut = int((1.0 - exclude_bottom_frac) * h) if exclude_bottom_frac > 0 else h
    for idx in range(1, count):
        area = int(stats[idx, cv2m.CC_STAT_AREA])
        x = int(stats[idx, cv2m.CC_STAT_LEFT])
        y = int(stats[idx, cv2m.CC_STAT_TOP])
        width = int(stats[idx, cv2m.CC_STAT_WIDTH])
        height = int(stats[idx, cv2m.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if height > max_bbox_height:
            continue
        if y + height >= bottom_cut:
            continue
        if x <= 1 or y <= 1 or x + width >= w - 2:
            continue
        keep[labels == idx] = 1
    return keep.astype(np.float32)


def clean_actor_alpha(cv2m, base_mask: np.ndarray, warped_anchor_mask: np.ndarray, dilate: int, blur: float) -> np.ndarray:
    raw = np.maximum(base_mask, warped_anchor_mask)
    binary = (raw > 0.18).astype(np.uint8)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        binary = cv2m.dilate(binary, kernel, iterations=1)
    alpha = binary.astype(np.float32)
    if blur > 0:
        alpha = cv2m.GaussianBlur(alpha, (0, 0), blur)
    return np.clip(alpha, 0.0, 1.0)


def component_validated_alpha(
    cv2m,
    npm,
    base: np.ndarray,
    warped_anchor: np.ndarray,
    warped_anchor_mask: np.ndarray,
    warped_valid: np.ndarray,
    min_area: int,
    max_area_frac: float,
    max_bbox_height_frac: float,
    exclude_bottom_frac: float,
    boundary: int,
    max_boundary_diff: float,
    min_valid_frac: float,
    dilate: int,
    blur: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    h, w = warped_anchor_mask.shape[:2]
    raw = (warped_anchor_mask > 0.18).astype(np.uint8)
    count, labels, stats, _ = cv2m.connectedComponentsWithStats(raw, connectivity=8)
    keep = np.zeros_like(raw)
    diagnostics: list[dict[str, float]] = []
    max_area = int(max_area_frac * h * w) if max_area_frac > 0 else h * w
    max_bbox_height = int(max_bbox_height_frac * h) if max_bbox_height_frac > 0 else h
    bottom_cut = int((1.0 - exclude_bottom_frac) * h) if exclude_bottom_frac > 0 else h + 1
    boundary_kernel = np.ones((max(1, boundary), max(1, boundary)), dtype=np.uint8)
    for idx in range(1, count):
        area = int(stats[idx, cv2m.CC_STAT_AREA])
        x = int(stats[idx, cv2m.CC_STAT_LEFT])
        y = int(stats[idx, cv2m.CC_STAT_TOP])
        width = int(stats[idx, cv2m.CC_STAT_WIDTH])
        height = int(stats[idx, cv2m.CC_STAT_HEIGHT])
        accepted = 0.0
        valid_frac = 0.0
        boundary_diff = 999.0
        if min_area <= area <= max_area and height <= max_bbox_height:
            touches_bad_region = y + height >= bottom_cut or x <= 1 or y <= 1 or x + width >= w - 2
            component = (labels == idx).astype(np.uint8)
            if not touches_bad_region:
                valid_frac = float(np.mean(warped_valid[component > 0] > 0.98))
                dilated = cv2m.dilate(component, boundary_kernel, iterations=1)
                ring = (dilated > 0) & (component == 0) & (warped_valid > 0.98)
                if npm.any(ring):
                    diff = npm.abs(base.astype(npm.float32) - warped_anchor.astype(npm.float32))
                    boundary_diff = float(npm.mean(diff[ring]))
                if valid_frac >= min_valid_frac and boundary_diff <= max_boundary_diff:
                    keep[component > 0] = 1
                    accepted = 1.0
        diagnostics.append(
            {
                "area": float(area),
                "height": float(height),
                "valid_frac": valid_frac,
                "boundary_diff": boundary_diff,
                "accepted": accepted,
            }
        )
    if dilate > 0:
        keep = cv2m.dilate(keep, np.ones((dilate, dilate), dtype=np.uint8), iterations=1)
    alpha = keep.astype(np.float32)
    if blur > 0:
        alpha = cv2m.GaussianBlur(alpha, (0, 0), blur)
    return np.clip(alpha, 0.0, 1.0), diagnostics


def color_match_paste_region(
    cv2m,
    npm,
    base: np.ndarray,
    warped_anchor: np.ndarray,
    alpha: np.ndarray,
    boundary: int,
    max_shift: float,
) -> np.ndarray:
    core = alpha > 0.05
    if not npm.any(core):
        return warped_anchor
    kernel = npm.ones((max(1, boundary), max(1, boundary)), dtype=npm.uint8)
    dilated = cv2m.dilate(core.astype(npm.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & (~core)
    if int(npm.sum(ring)) < 32:
        return warped_anchor
    base_mean = npm.mean(base[ring].astype(npm.float32), axis=0)
    warp_mean = npm.mean(warped_anchor[ring].astype(npm.float32), axis=0)
    shift = npm.clip(base_mean - warp_mean, -max_shift, max_shift)
    out = warped_anchor.astype(npm.float32).copy()
    out[core] += shift
    return npm.clip(out, 0, 255).astype(npm.uint8)


def erase_nearby_base_ghosts(
    cv2m,
    npm,
    base: np.ndarray,
    base_mask: np.ndarray,
    alpha: np.ndarray,
    mask_thresh: float,
    alpha_keep_thresh: float,
    near_radius: int,
    min_area: int,
    max_area_frac: float,
    inpaint_radius: float,
) -> tuple[np.ndarray, float]:
    core = alpha > alpha_keep_thresh
    if not npm.any(core):
        return base, 0.0
    h, w = base_mask.shape[:2]
    near_kernel = npm.ones((max(1, near_radius), max(1, near_radius)), dtype=npm.uint8)
    near = cv2m.dilate(core.astype(npm.uint8), near_kernel, iterations=1).astype(bool)
    raw = ((base_mask > mask_thresh) & (~core) & near).astype(npm.uint8)
    count, labels, stats, _ = cv2m.connectedComponentsWithStats(raw, connectivity=8)
    keep = npm.zeros_like(raw)
    max_area = int(max_area_frac * h * w) if max_area_frac > 0 else h * w
    for idx in range(1, count):
        area = int(stats[idx, cv2m.CC_STAT_AREA])
        if min_area <= area <= max_area:
            keep[labels == idx] = 1
    if not npm.any(keep):
        return base, 0.0
    # A tiny dilation avoids leaving colored fringes from the old ghost.
    keep = cv2m.dilate(keep, npm.ones((5, 5), dtype=npm.uint8), iterations=1)
    erased = cv2m.inpaint(base, (keep * 255).astype(npm.uint8), inpaint_radius, cv2m.INPAINT_TELEA)
    return erased, float(npm.mean(keep > 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--flow_scale", type=float, default=0.5)
    parser.add_argument("--actor_strength", type=float, default=0.85)
    parser.add_argument("--mask_dilate", type=int, default=17)
    parser.add_argument("--mask_blur", type=float, default=5.0)
    parser.add_argument("--segment_batch", type=int, default=4)
    parser.add_argument("--tag", default="mf_actorflow_h4")
    parser.add_argument("--dynamic_only", action="store_true")
    parser.add_argument("--dynamic_residual_thresh", type=float, default=2.5)
    parser.add_argument("--dynamic_min_area", type=int, default=80)
    parser.add_argument("--dynamic_max_area_frac", type=float, default=0.02)
    parser.add_argument("--dynamic_max_bbox_height_frac", type=float, default=0.22)
    parser.add_argument("--dynamic_exclude_bottom_frac", type=float, default=0.03)
    parser.add_argument("--component_validate", action="store_true")
    parser.add_argument("--component_min_area", type=int, default=80)
    parser.add_argument("--component_max_area_frac", type=float, default=0.035)
    parser.add_argument("--component_max_bbox_height_frac", type=float, default=0.30)
    parser.add_argument("--component_exclude_bottom_frac", type=float, default=0.02)
    parser.add_argument("--component_boundary", type=int, default=15)
    parser.add_argument("--component_max_boundary_diff", type=float, default=26.0)
    parser.add_argument("--component_min_valid_frac", type=float, default=0.92)
    parser.add_argument("--component_color_match", action="store_true")
    parser.add_argument("--color_match_boundary", type=int, default=19)
    parser.add_argument("--color_match_max_shift", type=float, default=18.0)
    parser.add_argument("--erase_base_ghosts", action="store_true")
    parser.add_argument("--erase_mask_thresh", type=float, default=0.35)
    parser.add_argument("--erase_alpha_keep_thresh", type=float, default=0.06)
    parser.add_argument("--erase_near_radius", type=int, default=95)
    parser.add_argument("--erase_min_area", type=int, default=80)
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
            flow_ba = dense_flow_base_to_anchor(cv2m, base, anchor, args.flow_scale)
            warped_anchor = warp_from_anchor(cv2m, anchor, flow_ba, border_value=(0, 0, 0))
            warped_mask = warp_from_anchor(cv2m, anchor_mask.astype(np.float32), flow_ba, border_value=0)
            if args.component_validate:
                warped_valid = warp_valid_mask(cv2m, anchor_mask.shape[:2], flow_ba)
                alpha, diagnostics = component_validated_alpha(
                    cv2m,
                    np,
                    base,
                    warped_anchor,
                    warped_mask,
                    warped_valid,
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
            elif args.dynamic_only:
                alpha = clean_actor_alpha(cv2m, np.zeros_like(base_mask), warped_mask, args.mask_dilate, args.mask_blur)
            else:
                alpha = clean_actor_alpha(cv2m, base_mask, warped_mask, args.mask_dilate, args.mask_blur)
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
                warped_anchor = color_match_paste_region(
                    cv2m,
                    np,
                    base_eff,
                    warped_anchor,
                    alpha,
                    args.color_match_boundary,
                    args.color_match_max_shift,
                )
            alpha_means.append(float(np.mean(alpha)))
            erase_means.append(erase_fraction)
            alpha = (alpha * args.actor_strength)[..., None]
            out = np.clip(base_eff.astype(np.float32) * (1.0 - alpha) + warped_anchor.astype(np.float32) * alpha, 0, 255)
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
