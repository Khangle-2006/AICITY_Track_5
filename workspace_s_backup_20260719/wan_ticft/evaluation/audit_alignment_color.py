"""Audit generated-video alignment and systemic color errors.

This is a diagnostic gate, not a submission-time oracle. It tests whether a
candidate is being penalized by a consistent temporal, spatial, or color-space
error before more modeling work is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_report(path: Path) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video = Path(row.get("video", ""))
            if row.get("status") in {"ok", "skipped"} and video.exists():
                videos[row["sample_id"]] = video
    return videos


def source_fps(row: dict[str, Any]) -> float:
    try:
        return float((row.get("frame_indices") or {}).get("fps") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def target_offsets(row: dict[str, Any]) -> list[int]:
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    target = indices.get("target") or []
    if not condition:
        return list(range(1, len(target) + 1))
    anchor = int(condition[-1])
    return [max(1, int(index) - anchor) for index in target]


def video_fps(cv2, path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps


def mapped_offsets(row: dict[str, Any], generated_fps: float, delta: int) -> list[int]:
    offsets = target_offsets(row)
    src_fps = source_fps(row)
    if src_fps > 0 and generated_fps > 0:
        offsets = [max(1, int(round(x * generated_fps / src_fps))) for x in offsets]
    return [max(0, x + delta) for x in offsets]


def read_video_frames(cv2, path: Path, indices: list[int]) -> list[Any] | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            cap.release()
            return None
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def read_targets(cv2, row: dict[str, Any]) -> list[Any] | None:
    paths = ((row.get("frames") or {}).get("target") or [])
    frames = [read_rgb(cv2, Path(path)) for path in paths]
    return None if not frames or any(frame is None for frame in frames) else frames


def resize_like(cv2, image, target):
    h, w = target.shape[:2]
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)


def translate(cv2, np, image, dx: int, dy: int):
    h, w = image.shape[:2]
    matrix = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def fit_affine_rgb(np, pairs: list[tuple[Any, Any]], samples_per_frame: int = 4096):
    xs = []
    ys = []
    rng = np.random.default_rng(7)
    for pred, target in pairs:
        pred = pred.reshape(-1, 3).astype(np.float32) / 255.0
        target = target.reshape(-1, 3).astype(np.float32) / 255.0
        count = min(samples_per_frame, len(pred))
        indices = rng.choice(len(pred), size=count, replace=False)
        xs.append(pred[indices])
        ys.append(target[indices])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    design = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
    weights, *_ = np.linalg.lstsq(design, y, rcond=None)
    return weights


def apply_affine_rgb(np, image, weights):
    x = image.astype(np.float32) / 255.0
    out = x @ weights[:3] + weights[3]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def score(cv2, np, pairs: list[tuple[Any, Any]]) -> tuple[float, float, float]:
    p = [psnr(cv2, np, pred, target) for pred, target in pairs]
    s = [ssim_gray(cv2, np, pred, target) for pred, target in pairs]
    d = [mean_absdiff(np, pred, target) for pred, target in pairs]
    return float(np.mean(p)), float(np.mean(s)), float(np.mean(d))


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print("\n" + title)
    print("-" * 88)
    print(f"{'variant':<28} {'domain':<30} {'N':>5} {'PSNR':>9} {'SSIM':>9} {'AbsDiff':>9}")
    for row in rows:
        print(
            f"{row['variant']:<28} {row['domain']:<30} {row['n']:>5} "
            f"{row['psnr']:>9.4f} {row['ssim']:>9.4f} {row['absdiff']:>9.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temporal_radius", type=int, default=3)
    parser.add_argument("--spatial_radius", type=int, default=4)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    manifest = read_manifest(args.manifest)
    if args.limit is not None:
        manifest = manifest[: args.limit]
    videos = read_report(args.report)

    samples = []
    for row_index, row in enumerate(manifest):
        sample_id = str(row.get("sample_id", ""))
        video = videos.get(sample_id)
        targets = read_targets(cv2, row)
        if video is None or targets is None:
            continue
        fps = video_fps(cv2, video)
        frames_by_delta = {}
        for delta in range(-args.temporal_radius, args.temporal_radius + 1):
            preds = read_video_frames(cv2, video, mapped_offsets(row, fps, delta))
            if preds is None or len(preds) != len(targets):
                continue
            frames_by_delta[delta] = [resize_like(cv2, pred, target) for pred, target in zip(preds, targets)]
        if 0 not in frames_by_delta:
            continue
        samples.append(
            {
                "fold": row_index % 2,
                "sample_id": sample_id,
                "domain": str(row.get("view_type", "unknown")),
                "targets": targets,
                "preds": frames_by_delta,
            }
        )

    variants: dict[str, list[tuple[str, Any, Any]]] = defaultdict(list)
    for sample in samples:
        for delta, preds in sample["preds"].items():
            for pred, target in zip(preds, sample["targets"]):
                variants[f"temporal_{delta:+d}"].append((sample["domain"], pred, target))
        for pred, target in zip(sample["preds"][0], sample["targets"]):
            variants["channel_bgr"].append((sample["domain"], pred[..., ::-1], target))
            for dx, dy in [
                (-args.spatial_radius, 0),
                (args.spatial_radius, 0),
                (0, -args.spatial_radius),
                (0, args.spatial_radius),
            ]:
                variants[f"spatial_dx{dx:+d}_dy{dy:+d}"].append(
                    (sample["domain"], translate(cv2, np, pred, dx, dy), target)
                )

    # Cross-fitted global RGB affine correction: fit on one half, score the other.
    for test_fold in (0, 1):
        train_pairs = []
        for sample in samples:
            if sample["fold"] != test_fold:
                train_pairs.extend(zip(sample["preds"][0], sample["targets"]))
        weights = fit_affine_rgb(np, train_pairs)
        for sample in samples:
            if sample["fold"] == test_fold:
                for pred, target in zip(sample["preds"][0], sample["targets"]):
                    variants["rgb_affine_crossfit"].append(
                        (sample["domain"], apply_affine_rgb(np, pred, weights), target)
                    )

    output_rows = []
    for variant, items in variants.items():
        domains = ["macro_all"] + sorted({domain for domain, _, _ in items})
        for domain in domains:
            selected = [(pred, target) for d, pred, target in items if domain == "macro_all" or d == domain]
            if not selected:
                continue
            p, s, d = score(cv2, np, selected)
            output_rows.append(
                {
                    "variant": variant,
                    "domain": domain,
                    "n": len(selected),
                    "psnr": p,
                    "ssim": s,
                    "absdiff": d,
                }
            )

    output_rows.sort(key=lambda row: (row["domain"], -row["psnr"], row["variant"]))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "domain", "n", "psnr", "ssim", "absdiff"])
        writer.writeheader()
        writer.writerows(output_rows)

    print_table("ALIGNMENT AND COLOR AUDIT", output_rows)
    macro = [row for row in output_rows if row["domain"] == "macro_all"]
    baseline = next(row for row in macro if row["variant"] == "temporal_+0")
    best = max(macro, key=lambda row: row["psnr"])
    print("\nDecision summary")
    print("-" * 88)
    print(f"samples: {len(samples)}")
    print(f"baseline: {baseline['psnr']:.4f} PSNR / {baseline['ssim']:.4f} SSIM")
    print(f"best: {best['variant']} | {best['psnr']:.4f} PSNR / {best['ssim']:.4f} SSIM")
    print(f"delta PSNR: {best['psnr'] - baseline['psnr']:+.4f}")
    print(f"output: {args.out_csv}")


if __name__ == "__main__":
    main()
