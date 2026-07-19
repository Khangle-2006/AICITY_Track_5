"""Mask-route low-frequency content from a cleaner scaffold branch.

This is a diagnostic branch for the Track 5 ghost artifact. The primary branch
keeps its high-frequency detail everywhere. A lower-detail donor branch is only
allowed to replace low-frequency content inside a temporal/donor disagreement
mask, so it can suppress translucent double images without globally adopting
the donor's soft look.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def sorted_pngs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)


def gray_absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.abs(ag - bg) / 255.0


def soft_norm(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / max(1e-6, hi - lo), 0.0, 1.0)


def build_mask(primary_frames: list[np.ndarray], donor: np.ndarray, idx: int, args) -> np.ndarray:
    cur = primary_frames[idx]
    prev = primary_frames[max(0, idx - 1)]
    nxt = primary_frames[min(len(primary_frames) - 1, idx + 1)]
    temporal = np.maximum(gray_absdiff(cur, prev), gray_absdiff(cur, nxt))
    branch = gray_absdiff(cur, donor)
    dark = np.maximum(
        cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32)
        - cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY).astype(np.float32),
        0,
    ) / 255.0
    mask = soft_norm(temporal, args.temporal_lo, args.temporal_hi) * soft_norm(branch, args.branch_lo, args.branch_hi)
    mask = np.maximum(mask, soft_norm(dark, args.dark_lo, args.dark_hi) * soft_norm(branch, args.branch_lo, args.branch_hi))
    if args.dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate))
        mask = cv2.dilate(mask, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), args.mask_blur)
    return np.clip(mask, 0.0, 1.0)


def route(args) -> None:
    primary_paths = sorted_pngs(args.primary_dir)
    donor_paths = sorted_pngs(args.donor_dir)
    if len(primary_paths) != len(donor_paths):
        raise RuntimeError(f"frame count mismatch: {len(primary_paths)} vs {len(donor_paths)}")
    primary_frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in primary_paths]
    donor_frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in donor_paths]
    if any(frame is None for frame in primary_frames + donor_frames):
        raise RuntimeError("failed reading frames")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    strengths = [float(v) for v in args.strengths.split(",")]
    for strength in strengths:
        out_sub = args.out_dir / f"s{strength:.2f}".replace(".", "p")
        out_sub.mkdir(parents=True, exist_ok=True)
        mask_sub = args.out_dir / f"mask_s{strength:.2f}".replace(".", "p")
        mask_sub.mkdir(parents=True, exist_ok=True)
        for idx, (path, primary, donor) in enumerate(zip(primary_paths, primary_frames, donor_frames)):
            mask = build_mask(primary_frames, donor, idx, args)
            primary_low = cv2.GaussianBlur(primary.astype(np.float32), (0, 0), args.low_sigma)
            donor_low = cv2.GaussianBlur(donor.astype(np.float32), (0, 0), args.low_sigma)
            primary_detail = primary.astype(np.float32) - primary_low
            m3 = (mask * strength)[..., None]
            mixed_low = primary_low * (1.0 - m3) + donor_low * m3
            out = np.clip(mixed_low + primary_detail, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_sub / path.name), out)
            cv2.imwrite(str(mask_sub / path.name), (mask * 255).astype(np.uint8))
        print(f"wrote {out_sub}", flush=True)


def audit(args) -> None:
    primary_paths = sorted_pngs(args.primary_dir)
    pred_dirs = [p for p in sorted(args.pred_dir.iterdir()) if p.is_dir() and not p.name.startswith("mask_")]
    mask_dirs = [args.pred_dir / ("mask_" + p.name) for p in pred_dirs]
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = [int(v) for v in args.frames_to_show.split(",")]
    rows = []
    for frame_id in frame_ids:
        cells = []
        source_dirs = [args.primary_dir, args.donor_dir] + pred_dirs + mask_dirs[:1]
        labels = ["primary", "donor"] + [p.name for p in pred_dirs] + ["mask"]
        for folder, label in zip(source_dirs, labels):
            image = cv2.imread(str(folder / f"{frame_id}.png"), cv2.IMREAD_COLOR)
            if image is None:
                image = np.zeros_like(cv2.imread(str(primary_paths[0]), cv2.IMREAD_COLOR))
            cv2.putText(image, f"{label} f{frame_id}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
            cells.append(image)
        rows.append(np.concatenate(cells, axis=1))
    cv2.imwrite(str(args.audit_dir / "splat_lowfreq_keyframes.jpg"), np.concatenate(rows, axis=0))
    for pred_dir in pred_dirs:
        frames = []
        for path in primary_paths:
            left = cv2.imread(str(path), cv2.IMREAD_COLOR)
            right = cv2.imread(str(pred_dir / path.name), cv2.IMREAD_COLOR)
            if left is None or right is None:
                continue
            combo = np.concatenate([left, right], axis=1)
            cv2.putText(combo, "primary", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(combo, pred_dir.name, (left.shape[1] + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
            frames.append(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB))
        imageio.mimsave(args.audit_dir / f"compare_{pred_dir.name}.mp4", frames, fps=args.fps, quality=8)
    print(f"wrote {args.audit_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    route_p = sub.add_parser("route")
    route_p.add_argument("--primary_dir", type=Path, required=True)
    route_p.add_argument("--donor_dir", type=Path, required=True)
    route_p.add_argument("--out_dir", type=Path, required=True)
    route_p.add_argument("--strengths", default="0.35,0.55,0.75")
    route_p.add_argument("--low_sigma", type=float, default=6.0)
    route_p.add_argument("--mask_blur", type=float, default=4.0)
    route_p.add_argument("--dilate", type=int, default=13)
    route_p.add_argument("--temporal_lo", type=float, default=0.025)
    route_p.add_argument("--temporal_hi", type=float, default=0.16)
    route_p.add_argument("--branch_lo", type=float, default=0.020)
    route_p.add_argument("--branch_hi", type=float, default=0.18)
    route_p.add_argument("--dark_lo", type=float, default=0.018)
    route_p.add_argument("--dark_hi", type=float, default=0.13)
    audit_p = sub.add_parser("audit")
    audit_p.add_argument("--primary_dir", type=Path, required=True)
    audit_p.add_argument("--donor_dir", type=Path, required=True)
    audit_p.add_argument("--pred_dir", type=Path, required=True)
    audit_p.add_argument("--audit_dir", type=Path, required=True)
    audit_p.add_argument("--frames_to_show", default="5,6,37")
    audit_p.add_argument("--fps", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "route":
        route(args)
    elif args.cmd == "audit":
        audit(args)


if __name__ == "__main__":
    main()
