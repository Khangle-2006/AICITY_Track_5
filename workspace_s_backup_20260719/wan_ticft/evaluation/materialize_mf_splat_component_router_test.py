"""Strict component router between MF and splat PNG trees for BDD test."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from baseline.frame_io import numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_report(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["source_folder"] if "source_folder" in row else row["output_name"]: row for row in csv.DictReader(handle)}


def local_highfreq(gray: np.ndarray) -> np.ndarray:
    return np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0))


def component_mask(base: np.ndarray, source: np.ndarray, cfg: dict[str, float]):
    height, width = base.shape[:2]
    diff = np.mean(np.abs(source.astype(np.float32) - base.astype(np.float32)), axis=2)
    raw = (diff > cfg["diff_thresh"]).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    keep = np.zeros_like(raw)
    max_area = int(cfg["max_area_frac"] * height * width)
    base_gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    base_hf = local_highfreq(base_gray)
    source_hf = local_highfreq(source_gray)
    ring_kernel = np.ones((11, 11), np.uint8)
    total = 0
    accepted = 0
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(cfg["min_area"]) or area > max_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if x <= 1 or y <= 1 or x + w >= width - 2 or y + h >= height - 2:
            continue
        total += 1
        comp = labels == idx
        ring = (cv2.dilate(comp.astype(np.uint8), ring_kernel, iterations=1) > 0) & (~comp)
        if not np.any(ring):
            continue
        boundary = float(np.mean(diff[ring]))
        base_detail = float(np.mean(base_hf[comp])) + 1e-6
        source_detail = float(np.mean(source_hf[comp]))
        if boundary <= cfg["max_boundary"] and source_detail >= cfg["min_hf_ratio"] * base_detail:
            keep[comp] = 1
            accepted += 1
    return keep.astype(np.float32), accepted, total


def paste(base: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask > 0.5):
        return base
    out = base.copy()
    out[mask > 0.5] = source[mask > 0.5]
    return out


def load_frames(directory: Path) -> list[np.ndarray]:
    frames = [read_rgb(cv2, path) for path in numeric_pngs(directory)]
    return [frame for frame in frames if frame is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--base_root", type=Path, required=True)
    parser.add_argument("--splat_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--diff_thresh", type=float, default=16.0)
    parser.add_argument("--max_boundary", type=float, default=18.0)
    parser.add_argument("--min_area", type=float, default=80.0)
    parser.add_argument("--max_area_frac", type=float, default=0.020)
    parser.add_argument("--min_hf_ratio", type=float, default=0.95)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2.setNumThreads(4)
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_csv(args.test_csv) if row.get("view_type") == args.view_type]
    cfg = {
        "diff_thresh": args.diff_thresh,
        "max_boundary": args.max_boundary,
        "min_area": args.min_area,
        "max_area_frac": args.max_area_frac,
        "min_hf_ratio": args.min_hf_ratio,
    }
    report = []
    diagnostics = []
    for position, row in enumerate(rows, start=1):
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        out_sample = args.out_dir / output_name
        existing = verify_sample_output(out_sample, target_n)
        if existing.ok and not args.clean:
            report.append(
                {
                    "output_name": output_name,
                    "sample_id": row["sample_id"],
                    "view_type": row["view_type"],
                    "target_N": target_n,
                    "frames_found": existing.frames_found,
                    "status": "ok",
                }
            )
            diagnostics.append(
                {
                    "output_name": output_name,
                    "accepted_components": "",
                    "total_components": "",
                    "mask_mean": "",
                }
            )
            print(f"[{position}/{len(rows)}] {output_name} | skip(existing)", flush=True)
            continue
        base_dir = args.base_root / output_name
        splat_dir = args.splat_root / output_name
        base_frames = load_frames(base_dir)
        splat_frames = load_frames(splat_dir)
        if len(base_frames) != target_n or len(splat_frames) != target_n:
            raise RuntimeError(f"{output_name}: base={len(base_frames)} splat={len(splat_frames)} target={target_n}")
        out_sample.mkdir(parents=True, exist_ok=True)
        accepted_total = 0
        component_total = 0
        mask_means = []
        written = 0
        for index, (base, splat) in enumerate(zip(base_frames, splat_frames)):
            if splat.shape[:2] != base.shape[:2]:
                splat = cv2.resize(splat, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
            mask, accepted, total = component_mask(base, splat, cfg)
            out = paste(base, splat, mask)
            written += int(write_rgb(cv2, out_sample / f"{index}.png", out))
            accepted_total += accepted
            component_total += total
            mask_means.append(float(mask.mean()))
        verify = verify_sample_output(out_sample, target_n)
        if not verify.ok:
            raise RuntimeError(f"{output_name}: verify failed {verify.notes}")
        report.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "view_type": row["view_type"],
                "target_N": target_n,
                "frames_found": written,
                "status": "ok" if written == target_n else "bad",
            }
        )
        diagnostics.append(
            {
                "output_name": output_name,
                "accepted_components": accepted_total,
                "total_components": component_total,
                "mask_mean": f"{float(np.mean(mask_means)):.6f}",
            }
        )
        print(f"[{position}/{len(rows)}] {output_name} | ok", flush=True)
    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output_name", "sample_id", "view_type", "target_N", "frames_found", "status"])
        writer.writeheader()
        writer.writerows(report)
    with (args.out_dir / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output_name", "accepted_components", "total_components", "mask_mean"])
        writer.writeheader()
        writer.writerows(diagnostics)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
