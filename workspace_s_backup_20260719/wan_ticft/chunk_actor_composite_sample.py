"""Composite only detected actors from raw chunk onto the base sample."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ACTOR_CLASSES = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck


def numeric_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"), key=lambda p: int(p.stem))


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def local_color_match(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    src_f = src.astype(np.float32)
    ref_f = ref.astype(np.float32)
    strong = mask[..., 0] > 0.15
    if strong.sum() < 32:
        return src
    s = src_f[strong]
    r = ref_f[strong]
    s_mean = s.mean(axis=0)
    r_mean = r.mean(axis=0)
    s_std = s.std(axis=0) + 1e-3
    r_std = r.std(axis=0) + 1e-3
    out = (src_f - s_mean) * (r_std / s_std) + r_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def detections(model: YOLO, frame: np.ndarray, min_area: int, min_y: float, max_area_frac: float):
    h, w = frame.shape[:2]
    result = model.predict(frame, imgsz=960, conf=0.18, iou=0.55, verbose=False)[0]
    if result.boxes is None:
        return []
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    seg_masks = None
    if result.masks is not None and result.masks.data is not None:
        seg_masks = result.masks.data.detach().cpu().numpy()
    dets = []
    for idx, cls in enumerate(classes):
        if cls not in ACTOR_CLASSES:
            continue
        x1, y1, x2, y2 = boxes[idx]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        center_y = (y1 + y2) * 0.5 / h
        if area < min_area or area > max_area_frac * h * w or center_y < min_y:
            continue
        if seg_masks is not None and idx < len(seg_masks):
            obj = cv2.resize(seg_masks[idx].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            obj = (obj > 0.35).astype(np.float32)
        else:
            obj = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(obj, (int(x1), int(y1)), (int(x2), int(y2)), 1.0, -1)
        dets.append({"cls": cls, "box": np.array([x1, y1, x2, y2], dtype=np.float32), "mask": obj})
    return dets


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(area_a + area_b - inter, 1e-6)


def center_dist_frac(a: np.ndarray, b: np.ndarray, h: int, w: int) -> float:
    ca = np.array([(a[0] + a[2]) * 0.5 / w, (a[1] + a[3]) * 0.5 / h])
    cb = np.array([(b[0] + b[2]) * 0.5 / w, (b[1] + b[3]) * 0.5 / h])
    return float(np.linalg.norm(ca - cb))


def make_actor_mask(model: YOLO, base: np.ndarray, chunk: np.ndarray, min_area: int, min_y: float, expand: int, max_area_frac: float) -> np.ndarray:
    h, w = chunk.shape[:2]
    chunk_dets = detections(model, chunk, min_area, min_y, max_area_frac)
    base_dets = detections(model, base, min_area, min_y, max_area_frac)
    mask = np.zeros((h, w), dtype=np.float32)
    for det in chunk_dets:
        matched = False
        for base_det in base_dets:
            if det["cls"] != base_det["cls"]:
                continue
            iou = box_iou(det["box"], base_det["box"])
            dist = center_dist_frac(det["box"], base_det["box"], h, w)
            if iou >= 0.08 or dist <= 0.08:
                matched = True
                break
        if not matched:
            continue
        obj = det["mask"]
        if expand > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))
            obj = cv2.dilate(obj, kernel)
        mask = np.maximum(mask, obj)
    mask = cv2.GaussianBlur(mask, (0, 0), 5.0)
    return np.clip(mask, 0.0, 1.0)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def write_video(path: Path, rows: list[tuple[str, list[np.ndarray]]], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(frames) for _, frames in rows)
    tile_w, tile_h = 512, 288
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tile_w * len(rows), tile_h))
    for i in range(n):
        tiles = []
        for name, frames in rows:
            tile = cv2.resize(frames[i], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            tiles.append(label(tile, f"{name} f{i}"))
        writer.write(cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_RGB2BGR))
    writer.release()


def write_sheet(path: Path, rows: list[tuple[str, list[np.ndarray]]], positions: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for name, frames in rows:
        tiles = []
        for pos in positions:
            pos = min(pos, len(frames) - 1)
            tile = cv2.resize(frames[pos], (384, 216), interpolation=cv2.INTER_AREA)
            tiles.append(label(tile, f"{name} f{pos}"))
        out_rows.append(np.concatenate(tiles, axis=1))
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(out_rows, axis=0), cv2.COLOR_RGB2BGR))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=Path, required=True)
    p.add_argument("--chunk_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model", default="yolov8n-seg.pt")
    p.add_argument("--min_area", type=int, default=900)
    p.add_argument("--min_y", type=float, default=0.18)
    p.add_argument("--expand", type=int, default=12)
    p.add_argument("--max_area_frac", type=float, default=0.08)
    p.add_argument("--fps", type=float, default=10.0)
    args = p.parse_args()

    model = YOLO(args.model)
    base_paths = numeric_pngs(args.base_dir)
    chunk_paths = numeric_pngs(args.chunk_dir)
    count = min(len(base_paths), len(chunk_paths))
    base_frames = [read_rgb(path) for path in base_paths[:count]]
    chunk_frames = [read_rgb(path) for path in chunk_paths[:count]]
    actor_frames = []
    mask_frames = []
    for idx, (base, chunk) in enumerate(zip(base_frames, chunk_frames)):
        if chunk.shape[:2] != base.shape[:2]:
            chunk = cv2.resize(chunk, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
        mask = make_actor_mask(model, base, chunk, args.min_area, args.min_y, args.expand, args.max_area_frac)[..., None]
        matched = local_color_match(chunk, base, mask)
        out = np.clip(base.astype(np.float32) * (1.0 - mask) + matched.astype(np.float32) * mask, 0, 255).astype(np.uint8)
        actor_frames.append(out)
        mask_vis = np.repeat((mask * 255).astype(np.uint8), 3, axis=2)
        mask_frames.append(mask_vis)
        write_rgb(args.out_dir / "actor_chunk" / f"{idx}.png", out)
    rows = [("base", base_frames), ("chunk", chunk_frames), ("mask", mask_frames), ("actor_chunk", actor_frames)]
    write_video(args.out_dir / "compare_actor_chunk.mp4", rows, args.fps)
    write_sheet(args.out_dir / "compare_actor_chunk_keyframes.jpg", rows, [7, 27, 38, 58])
    print(args.out_dir)


if __name__ == "__main__":
    main()
