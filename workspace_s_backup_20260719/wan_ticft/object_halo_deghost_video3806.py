from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


OUT_ROOT = Path("/workspace/experiments/wan_ticft/object_halo_deghost_video3806")
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


def choose_main_car(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    best = None
    for idx, box in enumerate(result.boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls not in VEHICLE_CLASSES or conf < 0.18:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2.0
        # Main black car in video3806: right half, medium/large, not the ego hood.
        if cx < 760 or y1 > 430 or y2 < 350 or w < 120 or h < 110 or w > 560 or h > 390:
            continue
        area = w * h
        score = conf + area * 1e-5 + max(0.0, cx - 760) * 5e-4
        if best is None or score > best["score"]:
            best = {
                "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                "mask": masks[idx] if masks is not None else None,
                "score": score,
                "conf": conf,
            }
    return best


def resize_mask(mask: np.ndarray | None, shape: tuple[int, int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    h, w = shape[:2]
    return np.clip(cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)


def temporal_unmix(base: np.ndarray, prev: np.ndarray, beta: float) -> np.ndarray:
    cur = base.astype(np.float32)
    delta = cur - prev.astype(np.float32)
    return np.clip(cur + beta * delta, 0, 255).astype(np.uint8)


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)
    variants = {
        "halo_anchor_a45": OUT_ROOT / "halo_anchor_a45" / "video3806",
        "halo_anchor_a65": OUT_ROOT / "halo_anchor_a65" / "video3806",
        "halo_unmix_a65": OUT_ROOT / "halo_unmix_a65" / "video3806",
    }
    for folder in variants.values():
        folder.mkdir(parents=True)
    debug_dir = OUT_ROOT / "debug_masks"
    debug_dir.mkdir()

    base_paths = numeric(list(BASE_DIR.glob("*.png")))[:59]
    anchor_paths = numeric(list(ANCHOR_DIR.glob("*.png")))[:59]
    model = YOLO("yolov8n-seg.pt")

    rows = []
    prev_base = None
    for i, path in enumerate(base_paths):
        base = cv2.imread(str(path))
        anchor = cv2.imread(str(anchor_paths[i]))
        if anchor.shape[:2] != base.shape[:2]:
            anchor = cv2.resize(anchor, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
        unmix = temporal_unmix(base, prev_base, 0.75) if prev_base is not None else base.copy()
        prev_base = base.copy()

        outputs = {name: base.copy() for name in variants}
        if i < 18:
            det = choose_main_car(model(str(path), verbose=False, imgsz=960)[0])
            if det is not None:
                h, w = base.shape[:2]
                x1, y1, x2, y2 = det["xyxy"]
                bw, bh = x2 - x1, y2 - y1
                ex1 = max(0, int(x1 - 0.28 * bw))
                ex2 = min(w, int(x2 + 0.30 * bw))
                ey1 = max(0, int(y1 - 0.20 * bh))
                ey2 = min(h, int(y2 + 0.28 * bh))

                car = resize_mask(det["mask"], base.shape)
                if car is None:
                    car = np.zeros((h, w), dtype=np.float32)
                    car[int(y1):int(y2), int(x1):int(x2)] = 1.0
                car_core = cv2.erode((car > 0.28).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
                car_wide = cv2.dilate((car > 0.18).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=1)
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                roi_mask[ey1:ey2, ex1:ex2] = 1
                halo = (roi_mask > 0).astype(np.float32) * (1.0 - car_core.astype(np.float32))
                # Prefer changing dark moving trails around the current car; keep the car body untouched.
                diff = np.mean(np.abs(base.astype(np.float32) - anchor.astype(np.float32)), axis=2)
                gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
                dark = np.clip((160.0 - gray) / 160.0, 0.0, 1.0)
                ghost = np.clip((diff - 8.0) / 42.0, 0.0, 1.0) * dark
                halo = halo * np.maximum(ghost, car_wide.astype(np.float32) * 0.32)
                halo = cv2.GaussianBlur(halo.astype(np.float32), (0, 0), 4.0)
                decay = 1.0 if i <= 10 else max(0.0, (18 - i) / 8.0)
                halo = np.clip(halo * decay, 0.0, 1.0)

                for name, strength, donor in [
                    ("halo_anchor_a45", 0.45, anchor),
                    ("halo_anchor_a65", 0.65, anchor),
                    ("halo_unmix_a65", 0.65, unmix),
                ]:
                    m = np.clip(halo * strength, 0.0, 0.82)[..., None]
                    outputs[name] = np.clip(donor.astype(np.float32) * m + base.astype(np.float32) * (1.0 - m), 0, 255).astype(np.uint8)
                cv2.imwrite(str(debug_dir / f"{i}_halo.png"), (halo * 255).astype(np.uint8))
                cv2.imwrite(str(debug_dir / f"{i}_car_core.png"), (car_core * 255).astype(np.uint8))
                rows.append({"frame": i, "x1": ex1, "y1": ey1, "x2": ex2, "y2": ey2, "conf": f"{det['conf']:.4f}"})

        for name, folder in variants.items():
            cv2.imwrite(str(folder / f"{i}.png"), outputs[name])

    with (OUT_ROOT / "detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "x1", "y1", "x2", "y2", "conf"])
        writer.writeheader()
        writer.writerows(rows)
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
