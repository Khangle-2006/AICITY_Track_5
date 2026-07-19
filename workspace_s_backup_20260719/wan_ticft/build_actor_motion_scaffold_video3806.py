from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ACTOR_CLASSES = {0, 1, 2, 3, 5, 7}
SAMPLE_ID = "test_bdd_pc_5k_bdd_video3806_phase2"


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


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def detections(model: YOLO, frame: np.ndarray, min_area: int, min_y: float, max_area_frac: float, max_height_frac: float):
    h, w = frame.shape[:2]
    result = model.predict(frame, imgsz=960, conf=0.16, iou=0.55, verbose=False)[0]
    if result.boxes is None:
        return []
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    masks = None
    if result.masks is not None and result.masks.data is not None:
        masks = result.masks.data.detach().cpu().numpy()
    out = []
    for idx, cls in enumerate(classes):
        if cls not in ACTOR_CLASSES:
            continue
        x1, y1, x2, y2 = boxes[idx]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        height = max(0.0, y2 - y1)
        cy = (y1 + y2) * 0.5 / h
        if area < min_area or area > max_area_frac * h * w or cy < min_y:
            continue
        if height > max_height_frac * h:
            continue
        if y2 > 0.92 * h and area > 0.025 * h * w:
            continue
        if masks is not None and idx < len(masks):
            mask = cv2.resize(masks[idx].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.32).astype(np.float32)
        else:
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1.0, -1)
        out.append({"cls": int(cls), "box": np.array([x1, y1, x2, y2], dtype=np.float32), "mask": mask})
    return out


def local_color_match(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    strong = mask[..., 0] > 0.15
    if strong.sum() < 64:
        return src
    src_f = src.astype(np.float32)
    ref_f = ref.astype(np.float32)
    s = src_f[strong]
    r = ref_f[strong]
    s_mean, r_mean = s.mean(axis=0), r.mean(axis=0)
    s_std, r_std = s.std(axis=0) + 1e-3, r.std(axis=0) + 1e-3
    out = (src_f - s_mean) * (r_std / s_std) + r_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def actor_mask(model: YOLO, frame: np.ndarray, args) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for det in detections(model, frame, args.min_area, args.min_y, args.max_area_frac, args.max_height_frac):
        obj = det["mask"]
        if args.expand > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.expand * 2 + 1, args.expand * 2 + 1))
            obj = cv2.dilate(obj, kernel, iterations=1)
        mask = np.maximum(mask, obj)
    if args.blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), args.blur)
    return np.clip(mask, 0.0, 1.0)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def make_compare(path: Path, rows: list[tuple[str, list[np.ndarray]]], fps: float) -> None:
    tile_w, tile_h = 448, 256
    n = min(len(frames) for _, frames in rows)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tile_w * len(rows), tile_h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for idx in range(n):
        tiles = []
        for name, frames in rows:
            tile = cv2.resize(frames[idx], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            tiles.append(label(tile, f"{name} f{idx}"))
        writer.write(cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--motion_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--min_area", type=int, default=650)
    parser.add_argument("--min_y", type=float, default=0.12)
    parser.add_argument("--max_area_frac", type=float, default=0.12)
    parser.add_argument("--max_height_frac", type=float, default=0.45)
    parser.add_argument("--expand", type=int, default=10)
    parser.add_argument("--blur", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    model = YOLO(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_paths = numeric_pngs(args.base_dir / "video3806")
    motion_paths = numeric_pngs(args.motion_dir / "video3806")
    count = min(len(base_paths), len(motion_paths))
    base_frames = [read_rgb(path) for path in base_paths[:count]]
    motion_frames = [read_rgb(path) for path in motion_paths[:count]]
    scaffold_frames = []
    mask_frames = []
    for idx, (base, motion) in enumerate(zip(base_frames, motion_frames)):
        if motion.shape[:2] != base.shape[:2]:
            motion = cv2.resize(motion, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
        mask = actor_mask(model, motion, args)[..., None]
        matched = local_color_match(motion, base, mask)
        weight = np.clip(args.alpha * mask, 0.0, 1.0)
        scaffold = np.clip(base.astype(np.float32) * (1.0 - weight) + matched.astype(np.float32) * weight, 0, 255).astype(np.uint8)
        scaffold_frames.append(scaffold)
        mask_frames.append(np.repeat((mask * 255).astype(np.uint8), 3, axis=2))
        write_rgb(args.out_dir / "scaffold_frames" / "video3806" / f"{idx}.png", scaffold)
    video_path = args.out_dir / "videos" / f"{SAMPLE_ID}.mp4"
    write_video(video_path, scaffold_frames, args.fps)
    write_video(args.out_dir / "mask_video.mp4", mask_frames, args.fps)
    make_compare(
        args.out_dir / "actor_motion_scaffold_compare.mp4",
        [("base_clean", base_frames), ("motion_teacher", motion_frames), ("mask", mask_frames), ("scaffold", scaffold_frames)],
        args.fps,
    )
    with (args.out_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status"])
        writer.writeheader()
        writer.writerow({"sample_id": SAMPLE_ID, "video": str(video_path), "status": "ok"})
    print(args.out_dir)


if __name__ == "__main__":
    main()
