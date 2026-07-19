"""Apply chunk high-frequency detail to one submission sample and build a compare video."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def numeric_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"), key=lambda p: int(p.stem))


def read_frames(path: Path, count: int | None = None) -> list[np.ndarray]:
    paths = numeric_pngs(path)
    if count is not None:
        paths = paths[:count]
    frames = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot read {p}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def blur(frame: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(frame.astype(np.float32), (0, 0), sigma)


def lap_strength(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    return cv2.GaussianBlur(lap, (0, 0), 1.0)


def restore_base_lowfreq(out: np.ndarray, base: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    corrected = out.astype(np.float32) - blur(out, sigma) + blur(base, sigma)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def detail_donor(
    base: np.ndarray,
    donor: np.ndarray,
    alpha: float = 0.55,
    sigma: float = 1.15,
    threshold: float = 18.0,
    require_sharper: bool = False,
    restore_luma: bool = False,
) -> np.ndarray:
    if donor.shape[:2] != base.shape[:2]:
        donor = cv2.resize(donor, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
    b = base.astype(np.float32)
    d = donor.astype(np.float32)
    detail = d - blur(d, sigma)
    base_low = blur(b, sigma * 1.4)
    donor_low = blur(d, sigma * 1.4)
    diff = np.mean(np.abs(base_low - donor_low), axis=2)
    mask = np.clip((threshold - diff) / threshold, 0.0, 1.0)
    if require_sharper:
        sharp_delta = lap_strength(donor) - lap_strength(base)
        sharp_mask = np.clip((sharp_delta + 2.0) / 10.0, 0.0, 1.0)
        mask *= sharp_mask
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2)[..., None]
    out = np.clip(b + alpha * detail * mask, 0, 255).astype(np.uint8)
    if restore_luma:
        out = restore_base_lowfreq(out, base)
    return out


def write_frames(path: Path, frames: list[np.ndarray]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(path / f"{i}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def write_video(path: Path, rows: list[tuple[str, list[np.ndarray]]], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(frames) for _, frames in rows)
    h, w = rows[0][1][0].shape[:2]
    tile_w, tile_h = 640, 360
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tile_w * len(rows), tile_h))
    for i in range(n):
        tiles = []
        for name, frames in rows:
            tile = cv2.resize(frames[i], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            tiles.append(label(tile, f"{name} f{i}"))
        frame = np.concatenate(tiles, axis=1)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_sheet(path: Path, rows: list[tuple[str, list[np.ndarray]]], positions: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for name, frames in rows:
        tiles = []
        for pos in positions:
            tile = cv2.resize(frames[min(pos, len(frames) - 1)], (384, 216), interpolation=cv2.INTER_AREA)
            tiles.append(label(tile, f"{name} f{pos}"))
        out_rows.append(np.concatenate(tiles, axis=1))
    sheet = np.concatenate(out_rows, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--donor_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    base = read_frames(args.base_dir)
    donor = read_frames(args.donor_dir, len(base))
    detail55 = [detail_donor(b, d) for b, d in zip(base, donor)]
    strict35 = [detail_donor(b, d, alpha=0.35, threshold=10.0, require_sharper=True, restore_luma=True) for b, d in zip(base, donor)]
    strict55 = [detail_donor(b, d, alpha=0.55, threshold=10.0, require_sharper=True, restore_luma=True) for b, d in zip(base, donor)]
    write_frames(args.out_dir / "detail_a055", detail55)
    write_frames(args.out_dir / "strict_a035", strict35)
    write_frames(args.out_dir / "strict_a055", strict55)
    rows = [("base", base), ("chunk", donor), ("detail55", detail55), ("strict35", strict35), ("strict55", strict55)]
    write_video(args.out_dir / "compare_base_chunk_detail55.mp4", rows, args.fps)
    write_sheet(args.out_dir / "compare_keyframes.jpg", rows, [5, 8, 37, min(58, len(base) - 1)])
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
