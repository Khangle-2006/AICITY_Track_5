from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from actor_box_motion_teacher_video3806 import Det, detect, list_frames, match_dets, paste_warped_actor, read_rgb, write_rgb


def read_manifest_row(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"empty manifest {path}")


def read_history_pair(manifest: Path) -> tuple[np.ndarray, np.ndarray]:
    row = read_manifest_row(manifest)
    condition = (row.get("frame_indices") or {}).get("condition") or []
    last = int(condition[-1])
    cap = cv2.VideoCapture(str(row["video_path"]))
    out = []
    for idx in [last - 1, last]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"cannot read history frame {idx}")
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out[0], out[1]


def dense_flow(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 31, 4, 7, 1.5, 0).astype(np.float32)


def dilated_union(dets: list[Det], shape: tuple[int, int], radius: int = 12) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for d in dets:
        mask = np.maximum(mask, (d.mask > 0.1).astype(np.uint8))
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def residuals_for_history(flow: np.ndarray, hist_dets: list[Det]) -> dict[int, np.ndarray]:
    actor_union = dilated_union(hist_dets, flow.shape[:2], radius=16)
    bg = flow[~actor_union]
    if len(bg) < 100:
        bg_med = np.array([0.0, 0.0], dtype=np.float32)
    else:
        bg_med = np.median(bg, axis=0).astype(np.float32)
    residuals: dict[int, np.ndarray] = {}
    for idx, det in enumerate(hist_dets):
        obj = flow[det.mask > 0.2]
        if len(obj) < 20:
            continue
        residual = np.median(obj, axis=0).astype(np.float32) - bg_med
        mag = float(np.linalg.norm(residual))
        if 0.15 <= mag <= 18.0:
            residuals[idx] = residual
    print(f"background median flow: {bg_med.tolist()}")
    for idx, r in residuals.items():
        print(f"history actor {idx}: cls={hist_dets[idx].cls} box={hist_dets[idx].box.round(1).tolist()} residual={r.round(3).tolist()}")
    return residuals


def fake_teacher_from_source(source: Det, offset: np.ndarray) -> Det:
    tbox = source.box.copy()
    tbox[[0, 2]] += float(offset[0])
    tbox[[1, 3]] += float(offset[1])
    return Det(source.cls, source.conf, tbox, source.mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed_dir", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--sample", default="video3806")
    parser.add_argument("--gain", type=float, nargs="+", default=[0.6, 1.0])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.16, 0.24])
    parser.add_argument("--cutoff", type=int, nargs="+", default=[12, 24])
    parser.add_argument("--count", type=int, default=59)
    args = parser.parse_args()

    seed_paths = list_frames(args.seed_dir, args.sample, args.count)
    seeds = [read_rgb(p) for p in seed_paths]
    h, w = seeds[0].shape[:2]

    prev, cur = read_history_pair(args.manifest)
    if cur.shape[:2] != (h, w):
        prev = cv2.resize(prev, (w, h), interpolation=cv2.INTER_AREA)
        cur = cv2.resize(cur, (w, h), interpolation=cv2.INTER_AREA)

    model = YOLO("yolov8n-seg.pt")
    hist_dets = detect(model, cur, conf=0.12)
    flow = dense_flow(prev, cur)
    residuals = residuals_for_history(flow, hist_dets)
    if not residuals:
        raise RuntimeError("no history residual actors found")

    seed_dets = [detect(model, frame, conf=0.14) for frame in seeds]
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_rgb(args.out_root / "history_cur.png", cur)
    write_rgb(args.out_root / "history_actor_mask.png", np.repeat((dilated_union(hist_dets, (h, w), radius=0).astype(np.uint8) * 255)[..., None], 3, axis=2))

    for gain in args.gain:
        for alpha in args.alpha:
            for cutoff in args.cutoff:
                label = f"g{int(gain * 100):03d}_a{int(alpha * 100):03d}_c{cutoff:02d}"
                out_dir = args.out_root / label / args.sample
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                for i, seed in enumerate(seeds):
                    frame = seed.copy()
                    pairs = match_dets(seed_dets[i], hist_dets, (h, w))
                    decay = max(0.0, 1.0 - i / max(1, cutoff + 1))
                    for source, hist in sorted(pairs, key=lambda p: p[0].area):
                        hist_idx = next((idx for idx, d in enumerate(hist_dets) if d is hist), None)
                        if hist_idx is None or hist_idx not in residuals:
                            continue
                        offset = residuals[hist_idx] * float(i + 1) * gain * decay
                        if np.linalg.norm(offset) < 1.0:
                            continue
                        teacher = fake_teacher_from_source(source, offset)
                        frame = paste_warped_actor(frame, seed, source, teacher, 1.0, alpha, False)
                    write_rgb(out_dir / f"{i}.png", frame)
    print(args.out_root)


if __name__ == "__main__":
    main()
