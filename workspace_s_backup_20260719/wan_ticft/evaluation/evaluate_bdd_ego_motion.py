"""Evaluate global ego-camera motion predictors on BDD validation samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.generators import compute_flow, warp_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from baseline.refiner_features import (
    base_scale_for_view,
    build_feature_config,
    make_refiner_arrays,
    refiner_in_channels,
)
from baseline.refiner_model import TinyResidualRefiner, import_torch


CONFIGS = {
    "repeat_last": {"kind": "repeat", "model": "repeat"},
    "region_norm_medium": {"kind": "region"},
    "affine_partial_t2": {"model": "partial", "max_time": 2.0, "anchor": 0.00, "roi": 0.85},
    "affine_partial_t4": {"model": "partial", "max_time": 4.0, "anchor": 0.00, "roi": 0.85},
    "affine_partial_t6": {"model": "partial", "max_time": 6.0, "anchor": 0.00, "roi": 0.85},
    "affine_partial_t4_a15": {"model": "partial", "max_time": 4.0, "anchor": 0.15, "roi": 0.85},
    "affine_full_t4": {"model": "affine", "max_time": 4.0, "anchor": 0.00, "roi": 0.85},
    "homography_t4": {"model": "homography", "max_time": 4.0, "anchor": 0.00, "roi": 0.85},
    "region_affine_g10": {"kind": "region_affine", "global_weight": 0.10},
    "region_affine_g20": {"kind": "region_affine", "global_weight": 0.20},
    "region_affine_g30": {"kind": "region_affine", "global_weight": 0.30},
    "region_affine_g40": {"kind": "region_affine", "global_weight": 0.40},
    "affine_residual_a15": {"kind": "affine_residual", "anchor": 0.15},
    "affine_residual_a35": {"kind": "affine_residual", "anchor": 0.35},
    "affine_residual_a50": {"kind": "affine_residual", "anchor": 0.50},
    "phase_router_v1": {"kind": "phase_router"},
}

FIELDS = [
    "sample_id",
    "phase",
    "candidate",
    "history_gap",
    "matches",
    "inlier_ratio",
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


def target_offsets(row: dict[str, Any], target_count: int) -> list[int]:
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    target = indices.get("target") or []
    if not condition or len(target) != target_count:
        return list(range(1, target_count + 1))
    last_condition = int(condition[-1])
    return [max(1, int(index) - last_condition) for index in target]


def resize_for_estimation(cv2, image, width: int):
    h, w = image.shape[:2]
    if width <= 0 or w <= width:
        return image, 1.0
    scale = width / float(w)
    return cv2.resize(image, (width, max(2, int(round(h * scale)))), interpolation=cv2.INTER_AREA), scale


def estimate_transform(cv2, np, prev, last, model: str, width: int, roi_bottom: float):
    prev_small, scale = resize_for_estimation(cv2, prev, width)
    last_small, _ = resize_for_estimation(cv2, last, width)
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY)
    last_gray = cv2.cvtColor(last_small, cv2.COLOR_RGB2GRAY)

    mask = np.zeros_like(prev_gray)
    bottom = max(1, min(mask.shape[0], int(round(mask.shape[0] * roi_bottom))))
    mask[:bottom] = 255
    points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=1500,
        qualityLevel=0.01,
        minDistance=8,
        mask=mask,
        blockSize=7,
    )
    if points is None or len(points) < 12:
        return np.eye(3, dtype=np.float64), 0, 0.0

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        last_gray,
        points,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    if tracked is None or status is None:
        return np.eye(3, dtype=np.float64), 0, 0.0

    valid = status.reshape(-1).astype(bool)
    src = points.reshape(-1, 2)[valid]
    dst = tracked.reshape(-1, 2)[valid]
    if len(src) < 12:
        return np.eye(3, dtype=np.float64), len(src), 0.0

    if model == "partial":
        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=3000, confidence=0.995
        )
        transform = np.eye(3, dtype=np.float64)
        if matrix is not None:
            transform[:2] = matrix
    elif model == "affine":
        matrix, inliers = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=3000, confidence=0.995
        )
        transform = np.eye(3, dtype=np.float64)
        if matrix is not None:
            transform[:2] = matrix
    else:
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.5, maxIters=3000, confidence=0.995)
        transform = matrix if matrix is not None else np.eye(3, dtype=np.float64)

    small_to_full = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    full_to_small = np.diag([scale, scale, 1.0])
    transform = small_to_full @ transform @ full_to_small
    inlier_ratio = float(inliers.mean()) if inliers is not None else 0.0
    return transform, len(src), inlier_ratio


def motion_scale(offset: int, history_gap: int, decay: float = 0.96, max_scale: float = 6.0) -> float:
    velocity = 1.0 / max(1, history_gap)
    scale = 0.0
    for _ in range(offset):
        scale = min(max_scale, scale + velocity)
        velocity *= decay
    return scale


def soft_motion_mask(cv2, np, flow):
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    mask = 1.0 / (1.0 + np.exp(-(mag - 1.5) / 0.75))
    mask = np.clip(mask * 0.85, 0.0, 0.85).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3.0, sigmaY=3.0)
    return np.clip(mask, 0.0, 0.85).astype(np.float32)


def region_predictions(cv2, np, prev, last, offsets, history_gap: int, width: int):
    flow = compute_flow(cv2, np, prev, last, width)
    mask = soft_motion_mask(cv2, np, flow)[:, :, None]
    last_float = last.astype(np.float32)
    predictions = []
    for offset in offsets:
        scale = motion_scale(offset, history_gap)
        flow_pred = warp_rgb(cv2, np, last, flow, scale).astype(np.float32)
        predictions.append(np.clip((1.0 - mask) * last_float + mask * flow_pred, 0, 255).astype(np.uint8))
    return predictions


def global_predictions(cv2, np, last, offsets, history_gap: int, transform):
    identity = np.eye(3, dtype=np.float64)
    delta = transform - identity
    h, w = last.shape[:2]
    predictions = []
    for offset in offsets:
        time = min(4.0, offset / float(max(1, history_gap)))
        predictions.append(
            cv2.warpPerspective(
                last,
                identity + delta * time,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        )
    return predictions


def predict(cv2, np, conditions, offsets, history_gap: int, config, width: int, phase: str = ""):
    last = conditions[-1]
    kind = config.get("kind", "global")
    if kind == "phase_router":
        if str(phase) in {"0", "4"}:
            config = {"kind": "affine_residual", "anchor": 0.35}
        else:
            config = {"kind": "region_affine", "global_weight": 0.20}
        kind = config["kind"]
    if kind == "repeat" or len(conditions) < 2:
        return [last] * len(offsets), 0, 0.0

    prev = conditions[-2]
    if kind == "region":
        return region_predictions(cv2, np, prev, last, offsets, history_gap, width), 0, 0.0

    transform, matches, inlier_ratio = estimate_transform(
        cv2, np, prev, last, config.get("model", "partial"), width, config.get("roi", 0.85)
    )
    if kind == "region_affine":
        region = region_predictions(cv2, np, prev, last, offsets, history_gap, width)
        global_preds = global_predictions(cv2, np, last, offsets, history_gap, transform)
        weight = config["global_weight"]
        predictions = [
            np.clip(
                (1.0 - weight) * local.astype(np.float32) + weight * global_pred.astype(np.float32),
                0,
                255,
            ).astype(np.uint8)
            for local, global_pred in zip(region, global_preds)
        ]
        return predictions, matches, inlier_ratio

    if kind == "affine_residual":
        h, w = last.shape[:2]
        aligned_prev = cv2.warpPerspective(
            prev,
            transform,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        residual_flow = compute_flow(cv2, np, aligned_prev, last, width)
        residual_mask = soft_motion_mask(cv2, np, residual_flow)[:, :, None]
        global_preds = global_predictions(cv2, np, last, offsets, history_gap, transform)
        predictions = []
        last_float = last.astype(np.float32)
        for offset, global_pred in zip(offsets, global_preds):
            scale = motion_scale(offset, history_gap)
            residual = warp_rgb(cv2, np, last, residual_flow, scale).astype(np.float32)
            local = (1.0 - residual_mask) * last_float + residual_mask * residual
            local_global = cv2.warpPerspective(
                np.clip(local, 0, 255).astype(np.uint8),
                np.eye(3, dtype=np.float64) + (transform - np.eye(3, dtype=np.float64))
                * min(4.0, offset / float(max(1, history_gap))),
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            anchor = config["anchor"]
            predictions.append(
                np.clip(
                    anchor * last_float + (1.0 - anchor) * local_global.astype(np.float32), 0, 255
                ).astype(np.uint8)
            )
        return predictions, matches, inlier_ratio

    identity = np.eye(3, dtype=np.float64)
    delta = transform - identity
    h, w = last.shape[:2]
    last_float = last.astype(np.float32)
    predictions = []
    for offset in offsets:
        time = min(config["max_time"], offset / float(max(1, history_gap)))
        future_transform = identity + delta * time
        warped = cv2.warpPerspective(
            last,
            future_transform,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        anchor = config["anchor"]
        if anchor > 0:
            warped = np.clip(
                anchor * last_float + (1.0 - anchor) * warped.astype(np.float32), 0, 255
            ).astype(np.uint8)
        predictions.append(warped)
    return predictions, matches, inlier_ratio


def load_refiner(checkpoint: Path):
    torch, _, _ = import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    args = SimpleNamespace(**ckpt.get("args", {}))
    cfg = build_feature_config(args)
    model = TinyResidualRefiner.build(
        in_channels=refiner_in_channels(cfg),
        width=int(args.model_width),
        residual_scale=float(args.residual_scale),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return torch, device, model, cfg, args


def refiner_predictions(cv2, np, refiner, conditions, offsets, history_gap: int, phase: str):
    torch, device, model, cfg, args = refiner
    frames = [
        cv2.resize(frame, (int(args.train_width), int(args.train_height)), interpolation=cv2.INTER_AREA)
        for frame in conditions
    ]
    output_height, output_width = conditions[-1].shape[:2]
    full_last = conditions[-1]
    full_prev = conditions[-2] if len(conditions) >= 2 else full_last
    full_flow = compute_flow(cv2, np, full_prev, full_last, cfg.flow_width)
    predictions = []
    with torch.no_grad():
        for offset in offsets:
            features, _, base = make_refiner_arrays(
                cv2,
                np,
                frames,
                frames[-1],
                "bdd",
                offset,
                cfg,
                history_gap,
                phase,
            )
            x = torch.from_numpy(features.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            base_t = torch.from_numpy(base.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(x, base_t)[0].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
            low_base = base.astype(np.float32) / 255.0
            residual = pred - low_base
            residual_full = cv2.resize(
                residual,
                (output_width, output_height),
                interpolation=cv2.INTER_LINEAR,
            )
            full_scale = base_scale_for_view("bdd", offset, cfg, history_gap)
            full_base = warp_rgb(cv2, np, full_last, full_flow, full_scale).astype(np.float32) / 255.0
            predictions.append(
                np.clip((full_base + residual_full) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            )
    return predictions


def combine_predictions(np, classical, refiner, offsets, kind: str):
    if kind.startswith("horizon_router_"):
        threshold = int(kind.rsplit("_", 1)[-1])
        return [
            classical_pred if offset <= threshold else refiner_pred
            for classical_pred, refiner_pred, offset in zip(classical, refiner, offsets)
        ]
    if kind.startswith("refiner_blend_"):
        alpha = int(kind.rsplit("_", 1)[-1]) / 100.0
        return [
            np.clip(
                (1.0 - alpha) * classical_pred.astype(np.float32)
                + alpha * refiner_pred.astype(np.float32),
                0,
                255,
            ).astype(np.uint8)
            for classical_pred, refiner_pred in zip(classical, refiner)
        ]
    return refiner


def score_sequence(cv2, np, predictions, targets):
    values = []
    for pred, target in zip(predictions, targets):
        if pred.shape[:2] != target.shape[:2]:
            pred = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)
        values.append(
            (
                psnr(cv2, np, pred, target),
                ssim_gray(cv2, np, pred, target),
                mean_absdiff(np, pred, target),
            )
        )
    return tuple(float(np.mean([item[i] for item in values])) for i in range(3))


def print_summary(np, rows):
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
        print(
            f"{candidate:<28} {len(items):>5} "
            f"{np.mean([float(x['psnr']) for x in items]):>10.4f} "
            f"{np.mean([float(x['ssim']) for x in items]):>10.4f} "
            f"{np.mean([float(x['absdiff']) for x in items]):>10.4f} "
            f"{winners.get(candidate, 0):>7}"
        )

    print("\nBest candidate by phase")
    print("-" * 76)
    phases = sorted({str(row["phase"]) for row in rows if row["status"] == "ok"})
    for phase in phases:
        phase_candidates = []
        for candidate in CONFIGS:
            items = [
                row
                for row in rows
                if row["candidate"] == candidate
                and str(row["phase"]) == phase
                and row["status"] == "ok"
            ]
            if items:
                phase_candidates.append(
                    (
                        float(np.mean([float(item["psnr"]) for item in items])),
                        float(np.mean([float(item["ssim"]) for item in items])),
                        candidate,
                    )
                )
        if phase_candidates:
            best_psnr, best_ssim, best_candidate = max(phase_candidates)
            print(
                f"phase {phase:<3} {best_candidate:<28} "
                f"PSNR={best_psnr:.4f} SSIM={best_ssim:.4f}"
            )

    oracle = []
    for items in by_sample.values():
        oracle.append(max(items, key=lambda item: float(item["psnr"])))
    if oracle:
        print("\nOracle ceiling")
        print("-" * 76)
        print(
            f"best per sample              {len(oracle):>5} "
            f"{np.mean([float(x['psnr']) for x in oracle]):>10.4f} "
            f"{np.mean([float(x['ssim']) for x in oracle]):>10.4f} "
            f"{np.mean([float(x['absdiff']) for x in oracle]):>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--estimate_width", type=int, default=640)
    parser.add_argument("--refiner_checkpoint", type=Path)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    refiner = None
    if args.refiner_checkpoint:
        refiner = load_refiner(args.refiner_checkpoint)
        CONFIGS.update(
            {
                "refiner_v2": {"kind": "refiner_v2"},
                "horizon_router_8": {"kind": "horizon_router_8"},
                "horizon_router_16": {"kind": "horizon_router_16"},
                "horizon_router_30": {"kind": "horizon_router_30"},
                "refiner_blend_25": {"kind": "refiner_blend_25"},
                "refiner_blend_50": {"kind": "refiner_blend_50"},
                "refiner_blend_75": {"kind": "refiner_blend_75"},
            }
        )
    rows = []
    manifest = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"]
    for row_index, row in enumerate(manifest, start=1):
        print(f"[{row_index}/{len(manifest)}] {row.get('sample_id', '')}", flush=True)
        paths = row.get("frames") or {}
        conditions = load_images(cv2, paths.get("condition") or [])
        targets = load_images(cv2, paths.get("target") or [])
        indices = (row.get("frame_indices") or {}).get("condition") or []
        history_gap = max(1, int(indices[-1]) - int(indices[-2])) if len(indices) >= 2 else 1
        offsets = target_offsets(row, len(targets))
        cached_refiner = None
        cached_classical = None
        for candidate, config in CONFIGS.items():
            if not conditions or not targets:
                p, s, d, matches, inlier_ratio = 0.0, 0.0, 0.0, 0, 0.0
                status, notes = "bad", "missing condition or target frames"
            else:
                kind = config.get("kind", "")
                if kind == "refiner_v2" or kind.startswith("horizon_router_") or kind.startswith("refiner_blend_"):
                    if cached_refiner is None:
                        cached_refiner = refiner_predictions(
                            cv2,
                            np,
                            refiner,
                            conditions,
                            offsets,
                            history_gap,
                            str(row.get("phase", "")),
                        )
                    if cached_classical is None:
                        cached_classical, _, _ = predict(
                            cv2,
                            np,
                            conditions,
                            offsets,
                            history_gap,
                            {"kind": "phase_router"},
                            args.estimate_width,
                            str(row.get("phase", "")),
                        )
                    preds = combine_predictions(np, cached_classical, cached_refiner, offsets, kind)
                    matches, inlier_ratio = 0, 0.0
                else:
                    preds, matches, inlier_ratio = predict(
                        cv2,
                        np,
                        conditions,
                        offsets,
                        history_gap,
                        config,
                        args.estimate_width,
                        str(row.get("phase", "")),
                    )
                p, s, d = score_sequence(cv2, np, preds, targets)
                status, notes = "ok", ""
            rows.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "phase": row.get("phase", ""),
                    "candidate": candidate,
                    "history_gap": history_gap,
                    "matches": matches,
                    "inlier_ratio": f"{inlier_ratio:.6f}",
                    "psnr": f"{p:.6f}" if status == "ok" else "",
                    "ssim": f"{s:.6f}" if status == "ok" else "",
                    "absdiff": f"{d:.6f}" if status == "ok" else "",
                    "status": status,
                    "notes": notes,
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print_summary(np, rows)
    print(f"output {args.out_csv}")


if __name__ == "__main__":
    main()
