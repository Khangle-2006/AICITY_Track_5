from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from actor_box_motion_teacher_video3806 import (
    Det,
    detect,
    list_frames,
    match_dets,
    paste_warped_actor,
    read_rgb,
    write_rgb,
)


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_folder(row: dict) -> str:
    value = row.get("source_folder") or row.get("output_name") or row.get("sample_id") or row.get("sample_name")
    if not value:
        raise RuntimeError("manifest row has no usable sample id")
    return str(value)


def sample_id(row: dict) -> str:
    return str(row.get("sample_id") or row.get("sample_name") or sample_folder(row))


def expected_count(row: dict, fallback: int) -> int:
    if fallback > 0:
        return fallback
    for key in ("target_N", "target_frames", "frames"):
        value = row.get(key)
        if value:
            try:
                return int(value)
            except Exception:
                pass
    target = (row.get("frames") or {}).get("target") or []
    if target:
        return len(target)
    return 59


def find_seed_frames(seed_root: Path, row: dict, count: int) -> tuple[str, list[Path]] | None:
    candidates = [
        sample_folder(row),
        sample_id(row),
        str(row.get("sample_name") or ""),
    ]
    # BDD LTX gates commonly use bare videoNNNN folders under frames/.
    for name in list(candidates):
        if "video" in name:
            idx = name.find("video")
            bare = name[idx:].split("_", 1)[0]
            candidates.append(bare)
    seen = []
    for name in candidates:
        if name and name not in seen:
            seen.append(name)
    for name in seen:
        folder = seed_root / name
        if folder.is_dir():
            n = expected_count(row, count)
            if count <= 0:
                existing = []
                i = 0
                while (folder / f"{i}.png").exists():
                    existing.append(folder / f"{i}.png")
                    i += 1
                if existing:
                    paths = existing[:n] if n > 0 else existing
                    return name, paths
            paths = [folder / f"{i}.png" for i in range(n)]
            if paths[0].exists():
                return name, paths
    return None


def read_last_two_from_video(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        raise RuntimeError(f"history video too short: {path}")
    out = []
    for idx in [total - 2, total - 1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"cannot read {path} frame {idx}")
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out[0], out[1]


def read_history_pair(row: dict, height: int, width: int, history_video_root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    if history_video_root is not None:
        path = history_video_root / f"{sample_id(row)}_history.mp4"
        if path.exists():
            prev, cur = read_last_two_from_video(path)
            if cur.shape[:2] != (height, width):
                prev = cv2.resize(prev, (width, height), interpolation=cv2.INTER_AREA)
                cur = cv2.resize(cur, (width, height), interpolation=cv2.INTER_AREA)
            return prev, cur

    condition = (row.get("frame_indices") or {}).get("condition") or []
    if len(condition) >= 2 and row.get("video_path"):
        indices = [int(condition[-2]), int(condition[-1])]
        cap = cv2.VideoCapture(str(row["video_path"]))
        out = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"cannot read {row.get('video_path')} frame {idx}")
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        prev, cur = out
    else:
        cond_paths = (row.get("frames") or {}).get("condition") or []
        if len(cond_paths) < 2:
            raise RuntimeError(f"no history for {sample_id(row)}")
        prev = read_rgb(Path(cond_paths[-2]))
        cur = read_rgb(Path(cond_paths[-1]))
    if cur.shape[:2] != (height, width):
        prev = cv2.resize(prev, (width, height), interpolation=cv2.INTER_AREA)
        cur = cv2.resize(cur, (width, height), interpolation=cv2.INTER_AREA)
    return prev, cur


def dense_flow(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 31, 4, 7, 1.5, 0).astype(np.float32)


def dilated_union(dets: list[Det], shape: tuple[int, int], radius: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for det in dets:
        mask = np.maximum(mask, (det.mask > 0.1).astype(np.uint8))
    if radius > 0 and mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def residuals_for_history(flow: np.ndarray, hist_dets: list[Det], min_mag: float, max_mag: float) -> dict[int, np.ndarray]:
    actor_union = dilated_union(hist_dets, flow.shape[:2], radius=16)
    bg = flow[~actor_union]
    bg_med = np.median(bg, axis=0).astype(np.float32) if len(bg) >= 100 else np.array([0.0, 0.0], dtype=np.float32)
    residuals: dict[int, np.ndarray] = {}
    for idx, det in enumerate(hist_dets):
        obj = flow[det.mask > 0.2]
        if len(obj) < 20:
            continue
        residual = np.median(obj, axis=0).astype(np.float32) - bg_med
        mag = float(np.linalg.norm(residual))
        if min_mag <= mag <= max_mag:
            residuals[idx] = residual
    return residuals


def fake_teacher(source: Det, offset: np.ndarray) -> Det:
    box = source.box.copy()
    box[[0, 2]] += float(offset[0])
    box[[1, 3]] += float(offset[1])
    return Det(source.cls, source.conf, box, source.mask)


def write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write video {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed_frames", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--gain", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.16)
    parser.add_argument("--cutoff", type=int, default=18)
    parser.add_argument("--count", type=int, default=0, help="0 uses target_N / available frames per sample.")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--min_mag", type=float, default=0.35)
    parser.add_argument("--max_mag", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--copy_when_skip", action="store_true")
    parser.add_argument("--history_video_root", type=Path, default=None)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    model = YOLO("yolov8n-seg.pt")
    frame_root = args.out_root / "frames"
    video_root = args.out_root / "videos"
    if frame_root.exists():
        shutil.rmtree(frame_root)
    frame_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)

    report_rows = []
    for row in rows:
        sid = sample_id(row)
        found = find_seed_frames(args.seed_frames, row, args.count)
        if found is None:
            print(f"skip {sid}: no seed frames")
            continue
        out_name, seed_paths = found
        seed_paths = [p for p in seed_paths if p.exists()]
        seeds = [read_rgb(p) for p in seed_paths]
        h, w = seeds[0].shape[:2]
        status = "ok"
        notes = []
        try:
            prev, cur = read_history_pair(row, h, w, args.history_video_root)
            hist_dets = detect(model, cur, conf=0.12)
            flow = dense_flow(prev, cur)
            residuals = residuals_for_history(flow, hist_dets, args.min_mag, args.max_mag)
            seed_dets = [detect(model, frame, conf=0.14) for frame in seeds]
            out_frames = []
            for i, seed in enumerate(seeds):
                frame = seed.copy()
                pairs = match_dets(seed_dets[i], hist_dets, (h, w))
                decay = max(0.0, 1.0 - i / max(1, args.cutoff + 1))
                for source, hist in sorted(pairs, key=lambda pair: pair[0].area):
                    hist_idx = next((idx for idx, det in enumerate(hist_dets) if det is hist), None)
                    if hist_idx is None or hist_idx not in residuals:
                        continue
                    offset = residuals[hist_idx] * float(i + 1) * args.gain * decay
                    if np.linalg.norm(offset) < 1.0:
                        continue
                    frame = paste_warped_actor(frame, seed, source, fake_teacher(source, offset), 1.0, args.alpha, False)
                out_frames.append(frame)
            notes.append(f"hist_dets={len(hist_dets)} residuals={len(residuals)}")
        except Exception as exc:
            if not args.copy_when_skip:
                print(f"skip {sid}: {exc}")
                continue
            out_frames = seeds
            status = "copied"
            notes.append(str(exc))

        out_dir = frame_root / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(out_frames):
            write_rgb(out_dir / f"{i}.png", frame)
        video_path = video_root / f"{sid}.mp4"
        write_video(video_path, out_frames, fps=args.fps)
        report_rows.append(
            {
                "sample_id": sid,
                "dataset_part": row.get("dataset_part", ""),
                "view_type": row.get("view_type", ""),
                "phase": row.get("phase", ""),
                "video": str(video_path),
                "status": status,
                "notes": ";".join(notes),
            }
        )
        print(f"{sid}: {status} {notes[-1] if notes else ''}")

    if not report_rows:
        raise RuntimeError("no samples processed")
    with (args.out_root / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)
    print(args.out_root)


if __name__ == "__main__":
    main()
