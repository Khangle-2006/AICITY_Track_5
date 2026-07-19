from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


OUT_ROOT = Path("/workspace/experiments/wan_ticft/object_traj_clean_car_video3806")
BASE_DIR = Path("/workspace/experiments/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate/video3806")
ANCHOR_DIR = Path("/workspace/experiments/wan_ticft/memory_motion_anchor_video3806_diverse_clamp/anchor_root/video3806")
VEHICLE_CLASSES = {2, 3, 5, 7}


def numeric(paths: list[Path]) -> list[Path]:
    keyed = []
    for path in paths:
        match = re.search(r"([0-9]+)(?!.*[0-9])", path.stem)
        if match:
            keyed.append((int(match.group(1)), path))
    return [path for _, path in sorted(keyed)]


def choose_target(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    best = None
    for index, box in enumerate(result.boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls not in VEHICLE_CLASSES or conf < 0.18:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1
        area = width * height
        cx = (x1 + x2) / 2.0
        if cx < 760 or y1 > 430 or y2 < 350 or width < 120 or height < 110 or width > 520 or height > 360:
            continue
        score = conf + 0.00001 * area + 0.0005 * max(0.0, cx - 760)
        if best is None or score > best["score"]:
            best = {
                "xyxy": np.array([x1, y1, x2, y2], np.float32),
                "conf": conf,
                "score": score,
                "mask": masks[index] if masks is not None else None,
            }
    return best


def fill_and_smooth(detections: list[dict | None]) -> list[np.ndarray | None]:
    boxes: list[np.ndarray | None] = []
    last = None
    for det in detections:
        if det is not None:
            last = det["xyxy"]
        boxes.append(None if last is None else last.copy())
    next_box = None
    for index in range(len(boxes) - 1, -1, -1):
        if detections[index] is not None:
            next_box = detections[index]["xyxy"]
        elif boxes[index] is None and next_box is not None:
            boxes[index] = next_box.copy()
    smooth: list[np.ndarray | None] = []
    for index, box in enumerate(boxes):
        if box is None:
            smooth.append(None)
            continue
        vals = [boxes[j] for j in range(max(0, index - 2), min(len(boxes), index + 3)) if boxes[j] is not None]
        smooth.append(np.mean(vals, axis=0).astype(np.float32))
    return smooth


def fill_detections(detections: list[dict | None]) -> list[dict | None]:
    out: list[dict | None] = []
    last = None
    for det in detections:
        if det is not None:
            last = det
        out.append(last)
    next_det = None
    for index in range(len(out) - 1, -1, -1):
        if detections[index] is not None:
            next_det = detections[index]
        elif out[index] is None:
            out[index] = next_det
    return out


def resize_mask(mask: np.ndarray | None, shape: tuple[int, int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    height, width = shape[:2]
    return np.clip(cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR), 0, 1)


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)
    frame_root = OUT_ROOT / "frame_root" / "video3806"
    mask_root = OUT_ROOT / "mask_debug"
    frame_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)

    base_paths = numeric(list(BASE_DIR.glob("*.png")))[:59]
    anchor_paths = numeric(list(ANCHOR_DIR.glob("*.png")))[:59]
    model = YOLO("yolov8n-seg.pt")

    base_det = []
    anchor_det = []
    for index in range(22):
        base_det.append(choose_target(model(str(base_paths[index]), verbose=False, imgsz=960)[0]))
        anchor_det.append(choose_target(model(str(anchor_paths[index]), verbose=False, imgsz=960)[0]))
    target_boxes = fill_and_smooth(base_det)
    source_dets = fill_detections(anchor_det)

    rows = []
    for index, path in enumerate(base_paths):
        base = cv2.imread(str(path))
        out = base.copy()
        if index < 18 and target_boxes[index] is not None and source_dets[index] is not None:
            anchor = cv2.imread(str(anchor_paths[index]))
            height, width = base.shape[:2]
            tx1, ty1, tx2, ty2 = target_boxes[index]
            tw, th = tx2 - tx1, ty2 - ty1
            tx1 = max(0, int(tx1 - 0.10 * tw))
            tx2 = min(width, int(tx2 + 0.08 * tw))
            ty1 = max(0, int(ty1 - 0.08 * th))
            ty2 = min(height, int(ty2 + 0.10 * th))

            sx1, sy1, sx2, sy2 = source_dets[index]["xyxy"]
            sw, sh = sx2 - sx1, sy2 - sy1
            sx1 = max(0, int(sx1 - 0.10 * sw))
            sx2 = min(width, int(sx2 + 0.08 * sw))
            sy1 = max(0, int(sy1 - 0.08 * sh))
            sy2 = min(height, int(sy2 + 0.10 * sh))
            if tx2 > tx1 + 8 and ty2 > ty1 + 8 and sx2 > sx1 + 8 and sy2 > sy1 + 8:
                crop = anchor[sy1:sy2, sx1:sx2]
                crop = cv2.resize(crop, (tx2 - tx1, ty2 - ty1), interpolation=cv2.INTER_CUBIC)
                source_mask = resize_mask(source_dets[index]["mask"], anchor.shape)
                if source_mask is not None:
                    source_mask = source_mask[sy1:sy2, sx1:sx2]
                    source_mask = cv2.resize(source_mask, (tx2 - tx1, ty2 - ty1), interpolation=cv2.INTER_LINEAR)
                else:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    source_mask = (gray < 130).astype(np.float32)
                source_mask = cv2.dilate(
                    (source_mask > 0.22).astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                    iterations=1,
                ).astype(np.float32)
                source_mask = cv2.GaussianBlur(source_mask, (0, 0), 4.0)
                strength = 0.95 if index <= 10 else max(0.0, 0.95 * (18 - index) / 8.0)
                mask = np.clip(source_mask * strength, 0, 0.95)[..., None]
                roi = out[ty1:ty2, tx1:tx2].astype(np.float32)
                out[ty1:ty2, tx1:tx2] = np.clip(crop.astype(np.float32) * mask + roi * (1 - mask), 0, 255).astype(np.uint8)
                cv2.imwrite(str(mask_root / f"{index}.png"), (source_mask * 255).astype(np.uint8))
                rows.append(
                    {
                        "frame": index,
                        "tx1": tx1,
                        "ty1": ty1,
                        "tx2": tx2,
                        "ty2": ty2,
                        "sx1": sx1,
                        "sy1": sy1,
                        "sx2": sx2,
                        "sy2": sy2,
                        "strength": f"{strength:.4f}",
                    }
                )
        cv2.imwrite(str(frame_root / f"{index}.png"), out)

    with (OUT_ROOT / "track.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "tx1", "ty1", "tx2", "ty2", "sx1", "sy1", "sx2", "sy2", "strength"])
        writer.writeheader()
        writer.writerows(rows)
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
