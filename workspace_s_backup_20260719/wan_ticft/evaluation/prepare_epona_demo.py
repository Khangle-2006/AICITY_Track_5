#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--condition_frames", type=int, default=10)
    parser.add_argument("--model_fps", type=float, default=5.0)
    parser.add_argument("--future_frames", type=int, default=1)
    parser.add_argument(
        "--orientation",
        choices=["native", "portrait_ccw"],
        default="native",
    )
    return parser.parse_args()


def load_row(path, sample_id):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("sample_id") == sample_id:
                return row
    raise ValueError(f"sample not found: {sample_id}")


def read_frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read frame {index}")
    return frame


def orient_frame(frame, mode):
    if mode == "portrait_ccw" and frame.shape[0] > frame.shape[1]:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def caption_speed_mps(prompt):
    match = re.search(r"(\d+(?:\.\d+)?)\s*km/h", prompt or "", re.IGNORECASE)
    if not match:
        return 0.0
    return float(match.group(1)) / 3.6


def main():
    args = parse_args()
    row = load_row(args.manifest, args.sample_id)
    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data" / args.sample_id
    reference_dir = out_dir / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(row["video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {row['video_path']}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    stride = max(1, round(source_fps / args.model_fps))
    end_frame = int(row["frame_indices"]["condition"][-1])
    history = [
        end_frame - stride * (args.condition_frames - 1 - i)
        for i in range(args.condition_frames)
    ]
    if history[0] < 0:
        raise RuntimeError(f"not enough history: first frame would be {history[0]}")

    for old in data_dir.glob("*.png"):
        old.unlink()
    for i, frame_index in enumerate(history):
        frame = orient_frame(read_frame(cap, frame_index), args.orientation)
        cv2.imwrite(str(data_dir / f"{i:06d}.png"), frame)
    # DemoDataset zips images with condition_frames + 1 poses. The final image
    # is only a loader-alignment pad; test_demo encodes the first ten images.
    cv2.imwrite(
        str(data_dir / f"{args.condition_frames:06d}.png"),
        orient_frame(read_frame(cap, history[-1]), args.orientation),
    )

    reference_indices = []
    for step in range(1, args.future_frames + 1):
        reference_index = end_frame + stride * step
        reference_indices.append(reference_index)
        frame = read_frame(cap, reference_index)
        cv2.imwrite(str(reference_dir / f"future_{step:03d}.png"), frame)
        if step == 1:
            cv2.imwrite(str(reference_dir / "next_5fps.png"), frame)
    cap.release()

    speed_mps = caption_speed_mps(row.get("prompt", ""))
    displacement = speed_mps / args.model_fps
    poses = np.zeros((args.condition_frames + 1, 2), dtype=np.float32)
    yaws = np.zeros((args.condition_frames + 1, 1), dtype=np.float32)
    poses[:, 0] = displacement
    np.save(data_dir / "pose.npy", poses)
    np.save(data_dir / "yaw.npy", yaws)

    metadata = {
        "sample_id": args.sample_id,
        "video_path": row["video_path"],
        "source_fps": source_fps,
        "model_fps": args.model_fps,
        "stride": stride,
        "history_indices": history,
        "loader_pad_index": history[-1],
        "reference_indices": reference_indices,
        "caption_speed_mps": speed_mps,
        "orientation": args.orientation,
        "pose_shape": list(poses.shape),
        "yaw_shape": list(yaws.shape),
    }
    with open(out_dir / "prepare_report.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
