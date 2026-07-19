"""Compute FID between real target frames and report videos at target offsets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from baseline.frame_io import read_rgb
from wan_ticft.evaluation.evaluate_video_reports import map_target_offsets, read_report, video_fps
from wan_ticft.evaluation.compute_real_fid import degrade


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_png(path: Path, rgb) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def gather_real(rows: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for row in rows:
        paths = (row.get("frames") or {}).get("target") or []
        if not paths:
            continue
        sid = row["sample_id"]
        readable = []
        for index, path in enumerate(paths):
            image = read_rgb(cv2, Path(path))
            if image is None:
                continue
            write_png(out_dir / f"{sid}__{index:04d}.png", image)
            readable.append(index)
        if readable:
            kept.append(row)
    return kept


def gather_candidate(rows: list[dict], report_path: Path, out_dir: Path, deg, rng) -> int:
    report = read_report(report_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        sid = row["sample_id"]
        video = report.get(sid)
        if video is None:
            continue
        offsets = map_target_offsets(row, video_fps(cv2, video))
        wanted = sorted(set(offsets))
        frame_by_index = {}
        cap = cv2.VideoCapture(str(video))
        if cap.isOpened() and wanted:
            wanted_set = set(wanted)
            for index in range(max(wanted) + 1):
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                if index in wanted_set:
                    frame_by_index[index] = bgr
            cap.release()
        target_paths = (row.get("frames") or {}).get("target") or []
        for target_index, frame_index in enumerate(offsets[: len(target_paths)]):
            bgr = frame_by_index.get(frame_index)
            if bgr is None:
                continue
            if deg is not None:
                bgr = degrade(bgr, *deg, rng)
            cv2.imwrite(str(out_dir / f"{sid}__{target_index:04d}.png"), bgr)
            written += 1
    return written


def fid(real_dir: Path, cand_dir: Path, batch_size: int):
    from pytorch_fid.fid_score import calculate_fid_given_paths
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return calculate_fid_given_paths([str(real_dir), str(cand_dir)], batch_size, device, 2048, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=REPORT_CSV")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--blur", type=float, default=0.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--jpeg_q", type=int, default=100)
    parser.add_argument("--downscale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import numpy as np

    deg = None
    if args.blur or args.noise or args.jpeg_q < 100 or args.downscale < 1.0:
        deg = (args.blur, args.noise, args.jpeg_q, args.downscale)
    rng = np.random.default_rng(args.seed)

    if args.work.exists():
        shutil.rmtree(args.work)
    args.work.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_manifest(args.manifest) if row.get("view_type") == args.view_type][: args.limit]
    rows = gather_real(rows, args.work / "real")
    print(f"real samples={len(rows)} frames={len(list((args.work / 'real').glob('*.png')))}", flush=True)

    results = {}
    for spec in args.candidate:
        name, raw_path = spec.split("=", 1)
        out_dir = args.work / name
        count = gather_candidate(rows, Path(raw_path), out_dir, deg, rng)
        value = fid(args.work / "real", out_dir, args.batch_size)
        results[name] = value
        print(f"  {name:20s} frames={count:4d} TARGET-FID={value:.3f}", flush=True)

    print("=== TARGET-OFFSET REAL-FID (lower=better) ===")
    for name, value in sorted(results.items(), key=lambda item: item[1]):
        print(f"  {name:20s} {value:.3f}")


if __name__ == "__main__":
    main()
