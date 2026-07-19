from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ACTOR_CLASSES = {0, 1, 2, 3, 5, 7}
VEHICLE_CLASSES = {1, 2, 3, 5, 7}


@dataclass
class Det:
    cls: int
    conf: float
    box: np.ndarray
    mask: np.ndarray

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def group(self) -> int:
        return 1 if self.cls in VEHICLE_CLASSES else self.cls


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def list_frames(root: Path, sample: str = "video3806", count: int = 59) -> list[Path]:
    direct = root / sample
    if direct.is_dir():
        return [direct / f"{i}.png" for i in range(count)]
    return [root / f"{i}.png" for i in range(count)]


def detect(model: YOLO, frame: np.ndarray, conf: float = 0.14) -> list[Det]:
    h, w = frame.shape[:2]
    result = model.predict(frame, imgsz=960, conf=conf, iou=0.55, verbose=False, device="cpu")[0]
    if result.boxes is None:
        return []
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    seg_masks = None
    if result.masks is not None and result.masks.data is not None:
        seg_masks = result.masks.data.detach().cpu().numpy()
    dets: list[Det] = []
    for i, cls in enumerate(classes):
        if cls not in ACTOR_CLASSES:
            continue
        x1, y1, x2, y2 = boxes[i]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < 220 or area > 0.15 * h * w:
            continue
        if y2 > 0.965 * h and area > 0.025 * h * w:
            continue
        if seg_masks is not None and i < len(seg_masks):
            mask = cv2.resize(seg_masks[i].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.28).astype(np.float32)
        else:
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1.0, -1)
        dets.append(Det(int(cls), float(scores[i]), boxes[i], mask))
    return dets


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1.0, aa + bb - inter)


def center_dist(a: np.ndarray, b: np.ndarray) -> float:
    ac = np.array([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5])
    bc = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5])
    return float(np.linalg.norm(ac - bc))


def match_dets(seed: list[Det], teacher: list[Det], shape: tuple[int, int]) -> list[tuple[Det, Det]]:
    h, w = shape
    used: set[int] = set()
    pairs: list[tuple[Det, Det]] = []
    for t in sorted(teacher, key=lambda d: d.area, reverse=True):
        best = None
        best_cost = 1e9
        t_diag = max(1.0, np.hypot(t.box[2] - t.box[0], t.box[3] - t.box[1]))
        for idx, s in enumerate(seed):
            if idx in used or s.group != t.group:
                continue
            dist = center_dist(s.box, t.box)
            max_dist = min(0.28 * np.hypot(h, w), 2.2 * t_diag)
            ov = iou(s.box, t.box)
            if dist > max_dist and ov < 0.03:
                continue
            scale = np.sqrt(max(1.0, t.area) / max(1.0, s.area))
            if scale < 0.45 or scale > 2.4:
                continue
            cost = dist / max_dist + (1.0 - ov) * 0.35 - min(s.conf, t.conf) * 0.05
            if cost < best_cost:
                best = idx
                best_cost = cost
        if best is not None:
            used.add(best)
            pairs.append((seed[best], t))
    return pairs


def dilate_blur(mask: np.ndarray, dilate: int = 6, blur: float = 2.0) -> np.ndarray:
    out = mask.astype(np.float32)
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        out = cv2.dilate((out > 0.05).astype(np.uint8), kernel, iterations=1).astype(np.float32)
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    return np.clip(out, 0.0, 1.0)


def paste_warped_actor(
    canvas: np.ndarray,
    seed_frame: np.ndarray,
    source: Det,
    teacher: Det,
    strength: float,
    alpha: float,
    do_inpaint: bool,
) -> np.ndarray:
    h, w = seed_frame.shape[:2]
    src_box = source.box.copy()
    dst_box = src_box * (1.0 - strength) + teacher.box * strength
    dst_box[[0, 2]] = np.clip(dst_box[[0, 2]], 0, w - 1)
    dst_box[[1, 3]] = np.clip(dst_box[[1, 3]], 0, h - 1)
    if dst_box[2] - dst_box[0] < 5 or dst_box[3] - dst_box[1] < 5:
        return canvas
    displacement = center_dist(src_box, dst_box)
    if displacement < 2.0:
        return canvas

    pad = 5
    sx1 = max(0, int(np.floor(src_box[0])) - pad)
    sy1 = max(0, int(np.floor(src_box[1])) - pad)
    sx2 = min(w, int(np.ceil(src_box[2])) + pad)
    sy2 = min(h, int(np.ceil(src_box[3])) + pad)
    dx1 = max(0, int(np.floor(dst_box[0])) - pad)
    dy1 = max(0, int(np.floor(dst_box[1])) - pad)
    dx2 = min(w, int(np.ceil(dst_box[2])) + pad)
    dy2 = min(h, int(np.ceil(dst_box[3])) + pad)
    if sx2 <= sx1 or sy2 <= sy1 or dx2 <= dx1 or dy2 <= dy1:
        return canvas

    src_crop = seed_frame[sy1:sy2, sx1:sx2]
    src_mask = source.mask[sy1:sy2, sx1:sx2]
    target_size = (dx2 - dx1, dy2 - dy1)
    actor = cv2.resize(src_crop, target_size, interpolation=cv2.INTER_LINEAR)
    actor_mask = cv2.resize(src_mask, target_size, interpolation=cv2.INTER_LINEAR)
    actor_mask = dilate_blur(actor_mask, dilate=2, blur=1.2)

    out = canvas
    if do_inpaint and displacement > 7.0:
        old_mask = dilate_blur(source.mask, dilate=7, blur=0.0)
        inpaint_mask = (old_mask * 255).astype(np.uint8)
        repaired = cv2.inpaint(cv2.cvtColor(out, cv2.COLOR_RGB2BGR), inpaint_mask, 3, cv2.INPAINT_TELEA)
        out = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

    region = out[dy1:dy2, dx1:dx2].astype(np.float32)
    weight = np.clip(actor_mask[..., None] * alpha, 0.0, 0.85)
    out[dy1:dy2, dx1:dx2] = np.clip(region * (1.0 - weight) + actor.astype(np.float32) * weight, 0, 255).astype(np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dir", type=Path, required=True)
    parser.add_argument("--teacher_dir", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--sample", default="video3806")
    parser.add_argument("--strength", type=float, nargs="+", default=[0.35, 0.55])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.35, 0.50])
    parser.add_argument("--inpaint", action="store_true")
    parser.add_argument("--count", type=int, default=59)
    args = parser.parse_args()

    model = YOLO("yolov8n-seg.pt")
    seed_paths = list_frames(args.seed_dir, args.sample, args.count)
    teacher_paths = list_frames(args.teacher_dir, args.sample, args.count)
    seeds = [read_rgb(p) for p in seed_paths]
    teachers = [read_rgb(p) for p in teacher_paths]
    h, w = seeds[0].shape[:2]
    teachers = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) if f.shape[:2] != (h, w) else f for f in teachers]

    detections = []
    for idx, (seed, teacher) in enumerate(zip(seeds, teachers)):
        sd = detect(model, seed)
        td = detect(model, teacher)
        pairs = match_dets(sd, td, (h, w))
        detections.append((sd, td, pairs))
        print(f"frame {idx:02d}: seed={len(sd)} teacher={len(td)} pairs={len(pairs)}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "strength", "alpha", "inpaint"])
        for strength in args.strength:
            for alpha in args.alpha:
                label = f"s{int(strength * 100):03d}_a{int(alpha * 100):03d}"
                out_dir = args.out_root / label / args.sample
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                for idx, seed in enumerate(seeds):
                    frame = seed.copy()
                    for source, teacher in sorted(detections[idx][2], key=lambda p: p[0].area):
                        frame = paste_warped_actor(frame, seed, source, teacher, strength, alpha, args.inpaint)
                    write_rgb(out_dir / f"{idx}.png", frame)
                writer.writerow([label, strength, alpha, int(args.inpaint)])
    print(args.out_root)


if __name__ == "__main__":
    main()
