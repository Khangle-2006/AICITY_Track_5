from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline.frame_io import read_rgb
from wan_ticft.evaluation.materialize_depth_layered_renderer import (
    PerceptionModels,
    load_full_video_frames,
    write_video,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_report(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok"}


def video_fps(path: Path, fallback: float = 30.0) -> float:
    if path.is_dir():
        return fallback
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
    cap.release()
    return fps if fps > 1.0 else fallback


def numeric_pngs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)


def load_base_source(path: Path) -> list[np.ndarray]:
    if path.is_dir():
        frames = [read_rgb(cv2, item) for item in numeric_pngs(path)]
        return [frame for frame in frames if frame is not None]
    return load_full_video_frames(cv2, path)


def condition_frames(row: dict, width: int, height: int) -> list[np.ndarray]:
    paths = (row.get("frames") or {}).get("condition") or []
    frames = [img for raw in paths if (img := read_rgb(cv2, Path(raw))) is not None]
    return [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]


def dense_flow_current_to_previous(current: np.ndarray, previous: np.ndarray, scale: float) -> np.ndarray:
    h, w = current.shape[:2]
    sw, sh = max(32, int(w * scale)), max(32, int(h * scale))
    cur = cv2.resize(current, (sw, sh), interpolation=cv2.INTER_AREA)
    prv = cv2.resize(previous, (sw, sh), interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(prv, cv2.COLOR_RGB2GRAY),
        None,
        0.5,
        4,
        31,
        4,
        7,
        1.5,
        0,
    ).astype(np.float32)
    if (sw, sh) != (w, h):
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= w / float(sw)
        flow[..., 1] *= h / float(sh)
    return flow


def decay_scale(frame_index: int, decay: float, max_scale: float) -> float:
    total = 0.0
    step = 1.0
    for _ in range(frame_index + 1):
        total += step
        step *= decay
    return min(total, max_scale)


def connected_actor_components(mask: np.ndarray, min_area: int, max_area_frac: float, max_height_frac: float):
    h, w = mask.shape[:2]
    binary = (mask > 0.22).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    max_area = int(max_area_frac * h * w)
    max_height = int(max_height_frac * h)
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        ww = int(stats[idx, cv2.CC_STAT_WIDTH])
        hh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area or hh > max_height:
            continue
        if x <= 1 or y <= 1 or x + ww >= w - 2 or y + hh >= h - 2:
            continue
        yield (labels == idx).astype(np.float32), (x, y, ww, hh), area


def shifted_component(anchor: np.ndarray, component: np.ndarray, dx: float, dy: float, dilate: int, blur: float):
    h, w = component.shape[:2]
    mat = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    shifted_img = cv2.warpAffine(anchor, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shifted_mask = cv2.warpAffine(component, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shifted_mask = np.clip(shifted_mask, 0.0, 1.0)
    if dilate > 0:
        shifted_mask = cv2.dilate((shifted_mask > 0.05).astype(np.uint8), np.ones((dilate, dilate), np.uint8), iterations=1).astype(np.float32)
    if blur > 0:
        shifted_mask = cv2.GaussianBlur(shifted_mask, (0, 0), blur)
    return shifted_img, np.clip(shifted_mask, 0.0, 1.0)


def color_match(base: np.ndarray, paste: np.ndarray, alpha: np.ndarray, max_shift: float) -> np.ndarray:
    core = alpha > 0.15
    if int(core.sum()) < 24:
        return paste
    ring = cv2.dilate(core.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1).astype(bool) & (~core)
    if int(ring.sum()) < 48:
        return paste
    shift = np.mean(base[ring].astype(np.float32), axis=0) - np.mean(paste[ring].astype(np.float32), axis=0)
    shift = np.clip(shift, -max_shift, max_shift)
    out = paste.astype(np.float32)
    out[core] += shift
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrapolate last observed actor components by history motion onto a base report.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--sample_id", action="append", default=[])
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--segment_batch", type=int, default=4)
    parser.add_argument("--flow_scale", type=float, default=0.5)
    parser.add_argument("--motion_strength", type=float, default=0.75)
    parser.add_argument("--max_velocity", type=float, default=18.0)
    parser.add_argument("--decay", type=float, default=0.72)
    parser.add_argument("--max_scale", type=float, default=3.2)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--fade_start", type=int, default=9999)
    parser.add_argument("--fade_end", type=int, default=10000)
    parser.add_argument("--fade_floor", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=7)
    parser.add_argument("--mask_blur", type=float, default=2.0)
    parser.add_argument("--min_area", type=int, default=60)
    parser.add_argument("--max_area_frac", type=float, default=0.035)
    parser.add_argument("--max_height_frac", type=float, default=0.28)
    parser.add_argument("--color_match", action="store_true")
    parser.add_argument("--tag", default="actor_motion")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2.setNumThreads(2)
    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    out_dir = args.out_root / args.tag
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    base_report = read_report(args.base_report)
    wanted = set(args.sample_id or [])
    rows = [
        row
        for row in read_jsonl(args.manifest)
        if row.get("view_type") == args.view_type and (not wanted or row.get("sample_id") in wanted or row.get("source_folder") in wanted)
    ]
    if args.limit:
        rows = rows[: args.limit]
    models = PerceptionModels(torch.device("cuda" if torch.cuda.is_available() else "cpu"), args.model_cache, depth_backend="dpt")
    report = []
    diagnostics = []
    for pos, row in enumerate(rows, start=1):
        sid = row["sample_id"]
        base_row = base_report.get(sid)
        if not base_row:
            continue
        base_path = Path(base_row.get("video") or base_row.get("source_dir") or "")
        bases = load_base_source(base_path)
        if not bases:
            continue
        h, w = bases[0].shape[:2]
        cond = condition_frames(row, w, h)
        if len(cond) < 2:
            continue
        previous, anchor = cond[-2], cond[-1]
        prev_mask, anchor_mask = models.actor_masks([previous, anchor], args.segment_batch)
        current_to_previous = dense_flow_current_to_previous(anchor, previous, args.flow_scale)
        velocity = -current_to_previous
        outputs = []
        comps = list(connected_actor_components(anchor_mask, args.min_area, args.max_area_frac, args.max_height_frac))
        used = 0
        for frame_idx, base in enumerate(bases):
            accum = np.zeros_like(base, dtype=np.float32)
            alpha_accum = np.zeros((h, w), dtype=np.float32)
            scale = decay_scale(frame_idx, args.decay, args.max_scale) * args.motion_strength
            if frame_idx <= args.fade_start:
                frame_alpha = args.alpha
            elif frame_idx >= args.fade_end:
                frame_alpha = args.alpha * args.fade_floor
            else:
                ratio = (frame_idx - args.fade_start) / max(1, args.fade_end - args.fade_start)
                frame_alpha = args.alpha * ((1.0 - ratio) + ratio * args.fade_floor)
            for component, bbox, area in comps:
                inside = component > 0.1
                if int(inside.sum()) < args.min_area:
                    continue
                v = np.median(velocity[inside], axis=0)
                mag = float(np.linalg.norm(v))
                if not np.isfinite(mag) or mag < 0.15:
                    continue
                if mag > args.max_velocity:
                    v = v * (args.max_velocity / mag)
                patch_img, patch_alpha = shifted_component(anchor, component, float(v[0] * scale), float(v[1] * scale), args.mask_dilate, args.mask_blur)
                if args.color_match:
                    patch_img = color_match(base, patch_img, patch_alpha, 16.0)
                a = patch_alpha * frame_alpha
                accum += patch_img.astype(np.float32) * a[..., None]
                alpha_accum = np.maximum(alpha_accum, a)
                used += 1
            alpha_accum = np.clip(alpha_accum, 0.0, 0.95)
            out = base.astype(np.float32) * (1.0 - alpha_accum[..., None]) + accum
            outputs.append(np.clip(out, 0, 255).astype(np.uint8))
        video = out_dir / "videos" / f"{sid}.avi"
        write_video(cv2, outputs, video, video_fps(base_path), prepend_guard=False)
        report.append({"sample_id": sid, "view_type": row.get("view_type", ""), "video": str(video), "status": "ok", "frames": len(outputs), "fps": f"{video_fps(base_path):.6f}"})
        diagnostics.append({"sample_id": sid, "components": len(comps), "used": used, "frames": len(outputs)})
        print(f"[{pos}/{len(rows)}] {sid} comps={len(comps)} used={used} frames={len(outputs)}", flush=True)
    with (out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "view_type", "video", "status", "frames", "fps"])
        writer.writeheader()
        writer.writerows(report)
    with (out_dir / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "components", "used", "frames"])
        writer.writeheader()
        writer.writerows(diagnostics)
    print(out_dir)


if __name__ == "__main__":
    main()
