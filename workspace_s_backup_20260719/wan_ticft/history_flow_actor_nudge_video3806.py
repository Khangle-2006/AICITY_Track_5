from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ACTOR_CLASSES = {0, 1, 2, 3, 5, 7}


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def read_manifest_row(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"empty manifest {path}")


def actor_mask(model: YOLO, frame: np.ndarray, expand: int, blur: float) -> np.ndarray:
    h, w = frame.shape[:2]
    result = model.predict(frame, imgsz=960, conf=0.16, iou=0.55, verbose=False, device="cpu")[0]
    mask = np.zeros((h, w), dtype=np.float32)
    if result.boxes is None:
        return mask
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    seg_masks = result.masks.data.detach().cpu().numpy() if result.masks is not None and result.masks.data is not None else None
    for idx, cls in enumerate(classes):
        if cls not in ACTOR_CLASSES:
            continue
        x1, y1, x2, y2 = boxes[idx]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < 350 or area > 0.12 * h * w:
            continue
        if y2 > 0.94 * h and area > 0.025 * h * w:
            continue
        if seg_masks is not None and idx < len(seg_masks):
            obj = cv2.resize(seg_masks[idx].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            obj = (obj > 0.30).astype(np.float32)
        else:
            obj = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(obj, (int(x1), int(y1)), (int(x2), int(y2)), 1.0, -1)
        mask = np.maximum(mask, obj)
    if expand > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))
        mask = cv2.dilate((mask > 0.1).astype(np.uint8), kernel, iterations=1).astype(np.float32)
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
    return np.clip(mask, 0.0, 1.0)


def dense_flow(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 31, 4, 7, 1.5, 0)
    return flow.astype(np.float32)


def warp_forward_like(frame: np.ndarray, flow: np.ndarray, scale: float, border=0) -> np.ndarray:
    h, w = frame.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx - flow[..., 0] * scale
    map_y = yy - flow[..., 1] * scale
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed_dir", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.16, 0.24])
    parser.add_argument("--cutoff", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--expand", type=int, default=8)
    parser.add_argument("--blur", type=float, default=3.0)
    args = parser.parse_args()

    row = read_manifest_row(args.manifest)
    condition = (row.get("frame_indices") or {}).get("condition") or []
    last = int(condition[-1])
    video_path = Path(row["video_path"])
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in [last - 1, last]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"cannot read history frame {idx}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    prev, cur = frames
    ref = read_rgb(Path((row.get("frames") or {}).get("condition")[-1]))
    if cur.shape[:2] != ref.shape[:2]:
        cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
        prev = cv2.resize(prev, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)

    model = YOLO("yolov8n-seg.pt")
    mask = actor_mask(model, cur, args.expand, args.blur)
    flow = dense_flow(prev, cur)
    if flow.shape[:2] != cur.shape[:2]:
        flow = cv2.resize(flow, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_LINEAR)
    seed_frames = [read_rgb(args.seed_dir / "video3806" / f"{i}.png") for i in range(59)]
    if seed_frames[0].shape[:2] != cur.shape[:2]:
        cur = cv2.resize(cur, (seed_frames[0].shape[1], seed_frames[0].shape[0]), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (seed_frames[0].shape[1], seed_frames[0].shape[0]), interpolation=cv2.INTER_LINEAR)
        flow = cv2.resize(flow, (seed_frames[0].shape[1], seed_frames[0].shape[0]), interpolation=cv2.INTER_LINEAR)

    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "settings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha", "cutoff"])
        for alpha in args.alpha:
            for cutoff in args.cutoff:
                label = f"a{int(alpha*100):03d}_c{cutoff:02d}"
                out = args.out_root / label / "video3806"
                if out.exists():
                    shutil.rmtree(out)
                out.mkdir(parents=True, exist_ok=True)
                for i, seed in enumerate(seed_frames):
                    if i <= cutoff:
                        warped = warp_forward_like(cur, flow, i + 1)
                        warped_mask = warp_forward_like(mask, flow, i + 1)[..., None]
                        decay = max(0.0, 1.0 - i / max(1, cutoff + 1))
                        weight = np.clip(alpha * decay * warped_mask, 0.0, 0.45)
                        frame = np.clip(seed.astype(np.float32) * (1 - weight) + warped.astype(np.float32) * weight, 0, 255).astype(np.uint8)
                    else:
                        frame = seed
                    write_rgb(out / f"{i}.png", frame)
                writer.writerow([alpha, cutoff])
    write_rgb(args.out_root / "history_actor_mask.png", np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2))
    print(args.out_root)


if __name__ == "__main__":
    main()
