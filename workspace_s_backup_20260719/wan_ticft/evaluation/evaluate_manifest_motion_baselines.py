"""Evaluate time-normalized motion baselines on a frame-materialized manifest."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.generators import compute_flow, make_region_motion_mask, warp_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.data.materialize_future_manifest import extract_frames
from wan_ticft.evaluation.evaluate_bdd_ego_motion import estimate_transform


CONFIGS = {
    "repeat_last": {
        "decay": 0.0,
        "max_scale": 0.0,
        "anchor_decay": 1.0,
        "min_anchor": 1.0,
        "region": False,
    },
    "flow_norm_light": {
        "decay": 0.92,
        "max_scale": 3.0,
        "anchor_decay": 0.995,
        "min_anchor": 0.70,
        "region": False,
    },
    "flow_norm_medium": {
        "decay": 0.96,
        "max_scale": 6.0,
        "anchor_decay": 0.990,
        "min_anchor": 0.45,
        "region": False,
    },
    "flow_norm_long": {
        "decay": 0.98,
        "max_scale": 10.0,
        "anchor_decay": 0.985,
        "min_anchor": 0.25,
        "region": False,
    },
    "region_norm_medium": {
        "decay": 0.96,
        "max_scale": 6.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": True,
    },
    "region_affine_g20_history": {
        "decay": 0.96,
        "max_scale": 6.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": True,
        "global_blend": 0.20,
    },
    "region_multi_history_median": {
        "decay": 0.96,
        "max_scale": 6.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": True,
        "multi_history_median": True,
    },
    "region_multi_history_recency": {
        "decay": 0.96,
        "max_scale": 6.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": True,
        "multi_history_recency": True,
    },
    "region_norm_long12": {
        "decay": 0.985,
        "max_scale": 12.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": True,
    },
    "region_norm_long24": {
        "decay": 0.995,
        "max_scale": 24.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": True,
    },
    "region_actor_guard25": {
        "decay": 0.96,
        "max_scale": 6.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": True,
        "actor_guard": 0.25,
    },
    "region_actor_guard50": {
        "decay": 0.96,
        "max_scale": 6.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": True,
        "actor_guard": 0.50,
    },
    "region_long24_actor_guard50": {
        "decay": 0.995,
        "max_scale": 24.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": True,
        "actor_guard": 0.50,
    },
    "region_mask_warped": {
        "decay": 0.985,
        "max_scale": 12.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": False,
        "region_mask_warped": True,
    },
    "actor_layer_median": {
        "decay": 0.96,
        "max_scale": 6.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": False,
        "actor_layer": True,
    },
    "actor_layer_filled": {
        "decay": 0.985,
        "max_scale": 12.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": False,
        "actor_filled": True,
    },
    "actor_layer_small_filled": {
        "decay": 0.985,
        "max_scale": 12.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": False,
        "actor_filled": True,
        "min_actor_area": 20.0,
        "max_actor_area_frac": 0.025,
    },
    "bg_locked_region_long24": {
        "decay": 0.995,
        "max_scale": 24.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": False,
        "bg_locked_actor": True,
        "base_candidate": "region_norm_long24",
        "min_actor_area": 20.0,
        "max_actor_area_frac": 0.025,
        "fg_color_tau": 10.0,
        "fg_color_high": 28.0,
        "source_color_tau": 7.0,
        "source_color_high": 22.0,
        "source_motion_tau": 0.35,
        "mask_blur": 1.0,
    },
    "bg_locked_actor_small": {
        "decay": 0.985,
        "max_scale": 12.0,
        "anchor_decay": 1.0,
        "min_anchor": 0.0,
        "region": False,
        "bg_locked_actor": True,
        "base_candidate": "actor_layer_small_filled",
        "min_actor_area": 20.0,
        "max_actor_area_frac": 0.025,
        "fg_color_tau": 6.0,
        "fg_color_high": 18.0,
        "source_color_tau": 7.0,
        "source_color_high": 22.0,
        "source_motion_tau": 0.35,
        "mask_blur": 1.0,
    },
}

FIELDS = [
    "sample_id",
    "dataset_part",
    "view_type",
    "phase",
    "candidate",
    "history_gap",
    "target_count",
    "psnr",
    "ssim",
    "absdiff",
    "status",
    "notes",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_images(cv2, paths: list[str]) -> list[Any]:
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def load_consecutive_source_targets(
    cv2,
    row: dict[str, Any],
    frame_count: int,
    target_height: int,
    target_width: int,
) -> list[Any]:
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    if not condition:
        return []
    start_frame = int(condition[-1]) + 1
    end_frame = start_frame + frame_count - 1

    _ = end_frame
    with tempfile.TemporaryDirectory(prefix="motion_eval_") as raw_tmp:
        output_dir = Path(raw_tmp) / "target"
        try:
            paths = extract_frames(
                Path(str(row.get("video_path", ""))),
                output_dir,
                start_frame=start_frame,
                frame_count=frame_count,
            )
        except RuntimeError:
            return []

        targets = []
        for path in paths:
            rgb = read_rgb(cv2, path)
            if rgb is None:
                continue
            if rgb.shape[:2] != (target_height, target_width):
                rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
            targets.append(rgb)
        return targets


def resize_like(cv2, pred, target):
    if pred.shape[:2] == target.shape[:2]:
        return pred
    return cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)


def score_sequence(cv2, np, preds: list[Any], targets: list[Any]) -> tuple[float, float, float]:
    metrics = []
    for pred, target in zip(preds, targets):
        pred = resize_like(cv2, pred, target)
        metrics.append(
            (
                psnr(cv2, np, pred, target),
                ssim_gray(cv2, np, pred, target),
                mean_absdiff(np, pred, target),
            )
        )
    return tuple(float(np.mean([values[i] for values in metrics])) for i in range(3))


def target_offsets(row: dict[str, Any], target_count: int, consecutive: bool) -> list[int]:
    if consecutive:
        return list(range(1, target_count + 1))
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    target = indices.get("target") or []
    if not condition or len(target) != target_count:
        return list(range(1, target_count + 1))
    last_condition = int(condition[-1])
    return [max(1, int(index) - last_condition) for index in target]


def weighted_median(np, values, weights):
    """Component-wise weighted median over the first axis."""
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    weight_shape = (len(weights),) + (1,) * (values.ndim - 1)
    broadcast_weights = np.broadcast_to(np.asarray(weights).reshape(weight_shape), values.shape)
    sorted_weights = np.take_along_axis(broadcast_weights, order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    cutoff = 0.5 * float(np.sum(weights))
    indices = np.argmax(cumulative >= cutoff, axis=0)[None, ...]
    return np.take_along_axis(sorted_values, indices, axis=0)[0]


def make_actor_guard_mask(cv2, np, conditions, flow):
    """Find foreground-ish moving regions where repeated warping causes smear."""
    last = conditions[-1]
    background = np.median(np.stack(conditions, axis=0).astype(np.float32), axis=0)
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    color_delta = np.mean(np.abs(last.astype(np.float32) - background), axis=2)
    seed = ((color_delta > 8.0) & (mag > 0.20)) | (color_delta > 24.0)
    mask = seed.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 <= area <= mask.shape[0] * mask.shape[1] * 0.12:
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    guard = cv2.GaussianBlur(filled.astype(np.float32) / 255.0, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return np.clip(guard, 0.0, 1.0).astype(np.float32)


def small_component_mask(cv2, np, seed, min_area: float, max_area_frac: float, blur: float):
    mask = seed.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    max_area = float(max_area_frac) * mask.shape[0] * mask.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    if blur > 0:
        filled = cv2.GaussianBlur(filled.astype(np.float32) / 255.0, (0, 0), sigmaX=blur, sigmaY=blur)
    else:
        filled = filled.astype(np.float32) / 255.0
    return np.clip(filled, 0.0, 1.0).astype(np.float32)


def predict(
    cv2,
    np,
    conditions: list[Any],
    offsets: list[int],
    history_gap: int,
    view_type: str,
    config: dict[str, Any],
    args,
):
    last = conditions[-1]
    if config["min_anchor"] >= 1.0 or len(conditions) < 2:
        return [last] * len(offsets)

    flow = compute_flow(cv2, np, conditions[-2], last, args.flow_width)
    if config.get("bg_locked_actor"):
        base_name = str(config.get("base_candidate", "region_norm_long24"))
        if base_name not in CONFIGS:
            raise KeyError(f"unknown bg_locked_actor base_candidate: {base_name}")
        base_frames = predict(
            cv2,
            np,
            conditions,
            offsets,
            history_gap,
            view_type,
            CONFIGS[base_name],
            args,
        )
        background = np.median(np.stack(conditions, axis=0).astype(np.float32), axis=0)
        mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        source_delta = np.mean(np.abs(last.astype(np.float32) - background), axis=2)
        source_seed = ((mag > float(config.get("source_motion_tau", 0.35))) & (
            source_delta > float(config.get("source_color_tau", 7.0))
        )) | (source_delta > float(config.get("source_color_high", 22.0)))
        source_mask = small_component_mask(
            cv2,
            np,
            source_seed,
            float(config.get("min_actor_area", 20.0)),
            float(config.get("max_actor_area_frac", 0.025)),
            float(config.get("mask_blur", 1.0)),
        )
        clean_background = (
            (1.0 - source_mask[:, :, None]) * last.astype(np.float32)
            + source_mask[:, :, None] * background
        )
        outputs = []
        for base in base_frames:
            if base.shape[:2] != clean_background.shape[:2]:
                base = cv2.resize(
                    base,
                    (clean_background.shape[1], clean_background.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            delta = np.mean(np.abs(base.astype(np.float32) - clean_background), axis=2)
            seed = (delta > float(config.get("fg_color_tau", 10.0))) | (
                delta > float(config.get("fg_color_high", 28.0))
            )
            fg_mask = small_component_mask(
                cv2,
                np,
                seed,
                float(config.get("min_actor_area", 20.0)),
                float(config.get("max_actor_area_frac", 0.025)),
                float(config.get("mask_blur", 1.0)),
            )
            pred = (1.0 - fg_mask[:, :, None]) * clean_background + fg_mask[:, :, None] * base.astype(np.float32)
            outputs.append(np.clip(pred, 0, 255).astype(np.uint8))
        return outputs

    if (config.get("multi_history_median") or config.get("multi_history_recency")) and len(conditions) >= 3:
        condition_indices = args.condition_indices
        velocities = []
        recency_weights = []
        for index in range(1, len(conditions)):
            gap = (
                int(condition_indices[index]) - int(condition_indices[index - 1])
                if len(condition_indices) == len(conditions)
                else 1
            )
            if gap <= 0:
                continue
            pair_flow = compute_flow(
                cv2, np, conditions[index - 1], conditions[index], args.flow_width
            )
            velocities.append(pair_flow / float(gap))
            # Keep newer pairs influential without letting the newest pair
            # outweigh all older evidence and collapse back to the baseline.
            recency_weights.append(float(1.5 ** (index - 1)))
        if velocities:
            stacked = np.stack(velocities, axis=0)
            if config.get("multi_history_recency"):
                flow = weighted_median(np, stacked, recency_weights).astype(np.float32)
            else:
                flow = np.median(stacked, axis=0).astype(np.float32)
            # Keep the validated region_norm_medium scale convention.
            flow *= float(max(1, history_gap))
    if config.get("region_mask_warped"):
        soft_mask, _ = make_region_motion_mask(cv2, np, flow, view_type, args)
        max_alpha = args.region_overhead_max_alpha if view_type == "wts_overhead_or_fixed_camera" else args.region_max_alpha
        binary = (soft_mask >= max_alpha * 0.55).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
        binary = cv2.dilate(binary, np.ones((5, 5), dtype=np.uint8), iterations=1)
        mask_rgb = np.repeat(binary[:, :, None], 3, axis=2)
        scale = 0.0
        velocity = 1.0 / max(1, history_gap)
        by_offset = {}
        wanted = set(offsets)
        for index in range(max(offsets)):
            scale = min(config["max_scale"], scale + velocity)
            velocity *= config["decay"]
            offset = index + 1
            if offset not in wanted:
                continue
            warped = warp_rgb(cv2, np, last, flow, scale).astype(np.float32)
            warped_mask = warp_rgb(cv2, np, mask_rgb, flow, scale)[:, :, 0].astype(np.float32) / 255.0
            warped_mask = cv2.GaussianBlur(warped_mask, (0, 0), sigmaX=0.8, sigmaY=0.8)
            warped_mask = np.clip(warped_mask, 0.0, 1.0)[:, :, None]
            pred = (1.0 - warped_mask) * last.astype(np.float32) + warped_mask * warped
            by_offset[offset] = np.clip(pred, 0, 255).astype(np.uint8)
        return [by_offset[offset] for offset in offsets]

    if config.get("actor_filled"):
        background = np.median(np.stack(conditions, axis=0).astype(np.float32), axis=0)
        mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        color_delta = np.mean(np.abs(last.astype(np.float32) - background), axis=2)
        seed = ((mag > 0.35) & (color_delta > 7.0)) | (color_delta > 22.0)
        mask = seed.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8))
        mask = cv2.dilate(mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        min_area = float(config.get("min_actor_area", 30.0))
        max_area = float(config.get("max_actor_area_frac", 1.0)) * mask.shape[0] * mask.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
        source_mask = cv2.GaussianBlur(filled.astype(np.float32) / 255.0, (0, 0), sigmaX=1.0)
        source_mask = np.clip(source_mask, 0.0, 1.0)
        clean_background = (
            (1.0 - source_mask[:, :, None]) * last.astype(np.float32)
            + source_mask[:, :, None] * background
        )
        scale = 0.0
        velocity = 1.0 / max(1, history_gap)
        by_offset = {}
        wanted = set(offsets)
        mask_rgb = np.repeat((source_mask[:, :, None] * 255.0).astype(np.uint8), 3, axis=2)
        for index in range(max(offsets)):
            scale = min(config["max_scale"], scale + velocity)
            velocity *= config["decay"]
            offset = index + 1
            if offset not in wanted:
                continue
            warped_actor = warp_rgb(cv2, np, last, flow, scale).astype(np.float32)
            destination_mask = warp_rgb(cv2, np, mask_rgb, flow, scale)[:, :, :1].astype(np.float32) / 255.0
            destination_mask = np.clip(destination_mask, 0.0, 1.0)
            pred = (1.0 - destination_mask) * clean_background + destination_mask * warped_actor
            by_offset[offset] = np.clip(pred, 0, 255).astype(np.uint8)
        return [by_offset[offset] for offset in offsets]

    if config.get("actor_layer"):
        background = np.median(np.stack(conditions, axis=0).astype(np.float32), axis=0)
        mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        mask = 1.0 / (1.0 + np.exp(-(mag - 0.75) / 0.35))
        mask = (mask > 0.55).astype(np.uint8) * 255
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=1.5)
        mask = np.clip(mask, 0.0, 0.98)
        scale = 0.0
        velocity = 1.0 / max(1, history_gap)
        by_offset = {}
        wanted = set(offsets)
        for index in range(max(offsets)):
            scale = min(config["max_scale"], scale + velocity)
            velocity *= config["decay"]
            offset = index + 1
            if offset not in wanted:
                continue
            warped = warp_rgb(cv2, np, last, flow, scale).astype(np.float32)
            warped_mask = warp_rgb(
                cv2,
                np,
                np.repeat((mask[:, :, None] * 255.0).astype(np.uint8), 3, axis=2),
                flow,
                scale,
            )[:, :, :1].astype(np.float32) / 255.0
            pred = (1.0 - warped_mask) * background + warped_mask * warped
            by_offset[offset] = np.clip(pred, 0, 255).astype(np.uint8)
        return [by_offset[offset] for offset in offsets]

    region_mask = None
    if config["region"]:
        region_mask, _ = make_region_motion_mask(cv2, np, flow, view_type, args)
    actor_guard = make_actor_guard_mask(cv2, np, conditions, flow) if config.get("actor_guard") else None

    global_transform = None
    if config.get("global_blend"):
        global_transform, _, _ = estimate_transform(
            cv2,
            np,
            conditions[-2],
            last,
            "partial",
            args.global_estimate_width,
            0.85,
        )

    scale = 0.0
    velocity = 1.0 / max(1, history_gap)
    last_float = last.astype(np.float32)
    by_offset = {}
    wanted = set(offsets)
    for index in range(max(offsets)):
        scale = min(config["max_scale"], scale + velocity)
        velocity *= config["decay"]
        offset = index + 1
        if offset not in wanted:
            continue
        warped = warp_rgb(cv2, np, last, flow, scale).astype(np.float32)
        if region_mask is not None:
            mask = region_mask[:, :, None]
            pred = (1.0 - mask) * last_float + mask * warped
        else:
            anchor = max(config["min_anchor"], config["anchor_decay"] ** offset)
            pred = anchor * last_float + (1.0 - anchor) * warped
        if actor_guard is not None:
            guard = actor_guard[:, :, None] * float(config["actor_guard"])
            pred = (1.0 - guard) * pred + guard * last_float
        if global_transform is not None:
            identity = np.eye(3, dtype=np.float64)
            time = min(4.0, offset / float(max(1, history_gap)))
            global_pred = cv2.warpPerspective(
                last,
                identity + (global_transform - identity) * time,
                (last.shape[1], last.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            weight = float(config["global_blend"])
            pred = (1.0 - weight) * pred + weight * global_pred.astype(np.float32)
        by_offset[offset] = np.clip(pred, 0, 255).astype(np.uint8)
    return [by_offset[offset] for offset in offsets]


def print_summary(np, rows: list[dict[str, Any]], title: str = "Overall") -> None:
    print()
    print(title)
    print("-" * 76)
    winners: dict[str, int] = {}
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "ok":
            by_sample.setdefault(row["sample_id"], []).append(row)
    for items in by_sample.values():
        winner = max(items, key=lambda item: float(item["psnr"]))
        winners[winner["candidate"]] = winners.get(winner["candidate"], 0) + 1

    print(f"{'Candidate':<28} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10} {'Wins':>7}")
    print("-" * 76)
    for candidate in CONFIGS:
        items = [row for row in rows if row["candidate"] == candidate and row["status"] == "ok"]
        if not items:
            continue
        p = np.mean([float(row["psnr"]) for row in items])
        s = np.mean([float(row["ssim"]) for row in items])
        d = np.mean([float(row["absdiff"]) for row in items])
        print(f"{candidate:<28} {len(items):>5} {p:>10.4f} {s:>10.4f} {d:>10.4f} {winners.get(candidate, 0):>7}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--view_type", default="")
    parser.add_argument(
        "--source_future_frames",
        type=int,
        default=0,
        help="Decode this many consecutive future frames directly from each source video.",
    )
    parser.add_argument("--flow_width", type=int, default=426)
    parser.add_argument("--global_estimate_width", type=int, default=640)
    parser.add_argument("--region_motion_tau", type=float, default=1.5)
    parser.add_argument("--region_motion_temp", type=float, default=0.75)
    parser.add_argument("--region_max_alpha", type=float, default=0.85)
    parser.add_argument("--region_overhead_max_alpha", type=float, default=0.45)
    parser.add_argument("--region_mask_blur", type=float, default=3.0)
    parser.add_argument("--limit_per_domain", type=int, default=0)
    parser.add_argument("--candidates", nargs="+", choices=tuple(CONFIGS), default=None)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows = []
    manifest_rows = read_jsonl(args.manifest)
    if args.view_type:
        manifest_rows = [row for row in manifest_rows if row.get("view_type") == args.view_type]
    if args.limit_per_domain:
        counts: dict[str, int] = {}
        limited = []
        for row in manifest_rows:
            view = str(row.get("view_type", ""))
            if counts.get(view, 0) >= args.limit_per_domain:
                continue
            counts[view] = counts.get(view, 0) + 1
            limited.append(row)
        manifest_rows = limited
    selected_configs = (
        {name: CONFIGS[name] for name in args.candidates}
        if args.candidates
        else CONFIGS
    )

    for row in manifest_rows:
        paths = row.get("frames") or {}
        conditions = load_images(cv2, paths.get("condition") or [])
        if args.source_future_frames > 0:
            if conditions:
                height, width = conditions[-1].shape[:2]
                targets = load_consecutive_source_targets(
                    cv2,
                    row,
                    args.source_future_frames,
                    height,
                    width,
                )
            else:
                targets = []
        else:
            targets = load_images(cv2, paths.get("target") or [])
        indices = (row.get("frame_indices") or {}).get("condition") or []
        history_gap = max(1, int(indices[-1]) - int(indices[-2])) if len(indices) >= 2 else 1
        args.condition_indices = indices
        view_type = str(row.get("view_type", ""))
        offsets = target_offsets(row, len(targets), args.source_future_frames > 0)

        for candidate, config in selected_configs.items():
            if not conditions or not targets:
                p, s, d = 0.0, 0.0, 0.0
                status = "bad"
                notes = "missing readable condition or target frames"
            elif args.source_future_frames > 0 and len(targets) != args.source_future_frames:
                p, s, d = 0.0, 0.0, 0.0
                status = "bad"
                notes = f"decoded {len(targets)}/{args.source_future_frames} source future frames"
            else:
                preds = predict(cv2, np, conditions, offsets, history_gap, view_type, config, args)
                p, s, d = score_sequence(cv2, np, preds, targets)
                status = "ok"
                notes = ""
            rows.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "dataset_part": row.get("dataset_part", ""),
                    "view_type": row.get("view_type", ""),
                    "phase": row.get("phase", ""),
                    "candidate": candidate,
                    "history_gap": history_gap,
                    "target_count": len(targets),
                    "psnr": f"{p:.6f}" if status == "ok" else "",
                    "ssim": f"{s:.6f}" if status == "ok" else "",
                    "absdiff": f"{d:.6f}" if status == "ok" else "",
                    "status": status,
                    "notes": notes,
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(np, rows)
    views = sorted({str(row["view_type"]) for row in rows})
    for view in views:
        print_summary(np, [row for row in rows if row["view_type"] == view], f"View: {view}")
    bad = [row for row in rows if row["status"] != "ok"]
    print(f"\nbad rows {len(bad)}")
    for row in bad[:10]:
        print(f"  {row['sample_id']} | {row['candidate']} | {row['notes']}")
    print(f"output {args.out_csv}")


if __name__ == "__main__":
    main()
