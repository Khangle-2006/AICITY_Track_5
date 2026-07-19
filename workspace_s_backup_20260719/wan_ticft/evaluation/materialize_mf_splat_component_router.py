"""Component-level router between MF base and forward-splat extrapolation.

The splat candidate reduces averaging blur but can introduce rubbery flow
artifacts. This router only copies connected components from splat into MF when
the component boundary is compatible with MF and the source is not less sharp.
It deliberately avoids broad soft alpha blending.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from wan_ticft.evaluation.evaluate_video_reports import read_report, target_offsets, video_fps
from wan_ticft.evaluation.materialize_depth_layered_renderer import write_video


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)


def local_highfreq(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)
    return np.abs(gray.astype(np.float32) - blur)


def parse_variant(value: str) -> tuple[str, dict[str, float]]:
    parts = value.split(":")
    if len(parts) != 7:
        raise ValueError(
            "variant must be label:diff_thresh:max_boundary:min_area:max_area_frac:min_hf_ratio:feather"
        )
    label = parts[0]
    return label, {
        "diff_thresh": float(parts[1]),
        "max_boundary": float(parts[2]),
        "min_area": float(parts[3]),
        "max_area_frac": float(parts[4]),
        "min_hf_ratio": float(parts[5]),
        "feather": float(parts[6]),
    }


def component_mask(base: np.ndarray, source: np.ndarray, cfg: dict[str, float]) -> tuple[np.ndarray, int, int]:
    height, width = base.shape[:2]
    diff = np.mean(np.abs(source.astype(np.float32) - base.astype(np.float32)), axis=2)
    raw = (diff > cfg["diff_thresh"]).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    keep = np.zeros_like(raw)
    total = 0
    accepted = 0
    max_area = int(cfg["max_area_frac"] * height * width)
    base_gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    base_hf = local_highfreq(base_gray)
    source_hf = local_highfreq(source_gray)
    ring_kernel = np.ones((11, 11), np.uint8)

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
        comp_u8 = comp.astype(np.uint8)
        ring = (cv2.dilate(comp_u8, ring_kernel, iterations=1) > 0) & (~comp)
        if not np.any(ring):
            continue
        boundary = float(np.mean(diff[ring]))
        base_detail = float(np.mean(base_hf[comp])) + 1e-6
        source_detail = float(np.mean(source_hf[comp]))
        if boundary <= cfg["max_boundary"] and source_detail >= cfg["min_hf_ratio"] * base_detail:
            keep[comp] = 1
            accepted += 1

    return keep.astype(np.float32), accepted, total


def paste(base: np.ndarray, source: np.ndarray, mask: np.ndarray, feather: float) -> np.ndarray:
    if not np.any(mask > 0):
        return base
    if feather > 0:
        alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        out = base.astype(np.float32) * (1.0 - alpha) + source.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    out = base.copy()
    out[mask > 0.5] = source[mask > 0.5]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--splat_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--target_offsets_only",
        action="store_true",
        help="Only evaluate/paste at validation target offsets; all other frames keep MF base.",
    )
    args = parser.parse_args()

    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == args.view_type][: args.limit]
    base_report = read_report(args.base_report)
    splat_report = read_report(args.splat_report)
    default_variants = [
        "strict:16:18:80:0.020:0.95:0",
        "medium:14:24:80:0.050:0.90:0",
        "medium_f1:14:24:80:0.050:0.90:1.0",
        "loose:12:30:60:0.080:0.85:0",
    ]
    variants = [parse_variant(value) for value in (args.variant or default_variants)]
    reports = {label: [] for label, _ in variants}
    diag = []
    with (args.out_root / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for pos, row in enumerate(rows, start=1):
        sid = row["sample_id"]
        base_path = base_report.get(sid)
        splat_path = splat_report.get(sid)
        if base_path is None or splat_path is None:
            print(f"[{pos}/{len(rows)}] {sid} | skip", flush=True)
            continue
        base_frames, fps = read_video(base_path)
        splat_frames, _ = read_video(splat_path)
        count = min(len(base_frames), len(splat_frames))
        if count <= 0:
            print(f"[{pos}/{len(rows)}] {sid} | skip(no frames)", flush=True)
            continue
        splat_frames = [resize_like(splat_frames[index], base_frames[index]) for index in range(count)]
        base_frames = base_frames[:count]
        for label, cfg in variants:
            out_frames = [frame.copy() for frame in base_frames]
            accepted_total = 0
            component_total = 0
            mask_means = []
            indices = target_offsets(row) if args.target_offsets_only else list(range(count))
            indices = [index for index in indices if 0 <= index < count]
            for frame_index in indices:
                base = base_frames[frame_index]
                splat = splat_frames[frame_index]
                mask, accepted, total = component_mask(base, splat, cfg)
                out_frames[frame_index] = paste(base, splat, mask, cfg["feather"])
                accepted_total += accepted
                component_total += total
                mask_means.append(float(mask.mean()))
            out_path = args.out_root / label / "videos" / f"{sid}.avi"
            write_video(cv2, out_frames, out_path, fps or video_fps(cv2, base_path) or 30.0, prepend_guard=False)
            reports[label].append({"sample_id": sid, "video": str(out_path), "status": "ok", "future_frames": len(out_frames)})
            diag.append({
                "sample_id": sid,
                "variant": label,
                "accepted_components": accepted_total,
                "total_components": component_total,
                "mask_mean": f"{float(np.mean(mask_means)):.6f}",
            })
        print(f"[{pos}/{len(rows)}] {sid} | ok", flush=True)

    for label, rows_out in reports.items():
        path = args.out_root / label / "report.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "future_frames"])
            writer.writeheader()
            writer.writerows(rows_out)
    with (args.out_root / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "variant", "accepted_components", "total_components", "mask_mean"])
        writer.writeheader()
        writer.writerows(diag)
    print(f"output={args.out_root}")


if __name__ == "__main__":
    main()
