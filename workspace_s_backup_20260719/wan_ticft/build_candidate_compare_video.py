from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def numeric_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"), key=lambda item: int(item.stem))


def read_frames(path: Path) -> list[np.ndarray]:
    frames = []
    for item in numeric_pngs(path):
        frame = cv2.imread(str(item), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot read {item}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def write_video(path: Path, rows: list[tuple[str, list[np.ndarray]]], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(len(frames) for _, frames in rows)
    tile_w, tile_h = 512, 288
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tile_w * len(rows), tile_h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame_index in range(count):
        tiles = []
        for name, frames in rows:
            tile = cv2.resize(frames[frame_index], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            tiles.append(label(tile, f"{name} f{frame_index}"))
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
    sheet = np.concatenate(out_rows, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("items", nargs="+", help="label=folder")
    args = parser.parse_args()

    rows = []
    for item in args.items:
        label_name, folder = item.split("=", 1)
        rows.append((label_name, read_frames(Path(folder) / args.sample)))
    write_video(args.out_dir / f"{args.sample}_candidate_compare.mp4", rows, args.fps)
    write_sheet(args.out_dir / f"{args.sample}_candidate_keyframes.jpg", rows, [5, 8, 37, 58])
    print(args.out_dir)


if __name__ == "__main__":
    main()
