from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from train_masked_pool_refiner import build_model, make_input_stack, to_tensor


def read_report(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok"}


def read_video_bgr(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames


def video_fps(path: Path, fallback: float = 30.0) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
    cap.release()
    return fps if fps > 1.0 else fallback


def resize_to(frames: list[np.ndarray], width: int, height: int) -> list[np.ndarray]:
    out = []
    for frame in frames:
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        out.append(frame)
    return out


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply masked_pool_refiner checkpoint to a video report pool.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--medium_report", type=Path, required=True)
    parser.add_argument("--long_report", type=Path, required=True)
    parser.add_argument("--actor_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--strengths", default="0.35,0.55,0.75,1.00")
    parser.add_argument("--mask_thresh", type=float, default=0.0)
    parser.add_argument("--mask_gamma", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", default="masked_pool")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cv2.setNumThreads(2)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    width = int(train_args.get("model_width", 24))
    residual_scale = float(train_args.get("residual_scale", 0.24))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch, nn, F, width, residual_scale).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    medium = read_report(args.medium_report)
    long = read_report(args.long_report)
    actor = read_report(args.actor_report)
    sample_ids = [sid for sid in medium if sid in long and sid in actor]
    if args.limit:
        sample_ids = sample_ids[: args.limit]
    strengths = [float(item) for item in args.strengths.split(",")]
    reports = {strength: [] for strength in strengths}

    for pos, sid in enumerate(sample_ids, start=1):
        medium_path = Path(medium[sid]["video"])
        medium_frames = read_video_bgr(medium_path)
        height, width_px = medium_frames[0].shape[:2]
        count = len(medium_frames)
        long_frames = resize_to(read_video_bgr(Path(long[sid]["video"]))[:count], width_px, height)
        actor_frames = resize_to(read_video_bgr(Path(actor[sid]["video"]))[:count], width_px, height)
        medium_frames = medium_frames[: min(count, len(long_frames), len(actor_frames))]
        long_frames = long_frames[: len(medium_frames)]
        actor_frames = actor_frames[: len(medium_frames)]
        pred_frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for idx in range(len(medium_frames)):
            stack_np, center_np, infer = make_input_stack(medium_frames, long_frames, actor_frames, idx)
            x = to_tensor(torch, stack_np)[None].to(device)
            center = to_tensor(torch, center_np)[None].to(device)
            mask = torch.from_numpy(infer.astype(np.float32))[None, None].to(device)
            with torch.no_grad():
                pred, _ = model(x, center, mask)
            pred_np = (pred[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pred_frames.append(pred_np)
            masks.append(infer)

        fps = video_fps(medium_path)
        for strength in strengths:
            label = f"{args.tag}_s{strength:.2f}".replace(".", "p")
            frames = []
            for center_np, pred_np, infer in zip(medium_frames, pred_frames, masks):
                mask_eff = np.clip((infer - args.mask_thresh) / max(1e-6, 1.0 - args.mask_thresh), 0.0, 1.0)
                if args.mask_gamma != 1.0:
                    mask_eff = np.power(mask_eff, args.mask_gamma)
                mask3 = np.clip(mask_eff[..., None] * strength, 0.0, 1.0)
                blended = center_np.astype(np.float32) * (1.0 - mask3) + pred_np.astype(np.float32) * mask3
                frames.append(np.clip(blended, 0, 255).astype(np.uint8))
            out_path = args.out_root / label / "videos" / f"{sid}.avi"
            write_video(out_path, frames, fps)
            reports[strength].append(
                {
                    "sample_id": sid,
                    "view_type": medium[sid].get("view_type", ""),
                    "video": str(out_path),
                    "status": "ok",
                    "frames": len(frames),
                    "fps": f"{fps:.6f}",
                }
            )
        print(f"[{pos}/{len(sample_ids)}] {sid} frames={len(medium_frames)}", flush=True)

    for strength, rows in reports.items():
        label = f"{args.tag}_s{strength:.2f}".replace(".", "p")
        path = args.out_root / label / "report.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "view_type", "video", "status", "frames", "fps"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"report={path}", flush=True)


if __name__ == "__main__":
    main()
