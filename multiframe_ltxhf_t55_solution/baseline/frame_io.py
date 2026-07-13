from __future__ import annotations

import shutil
from pathlib import Path


def import_cv2_np():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        return cv2, np
    except Exception as exc:
        raise SystemExit(f"OpenCV and numpy are required. Original error: {exc}")


def numeric_pngs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    files = [p for p in input_dir.glob("*.png") if p.stem.isdigit()]
    return sorted(files, key=lambda p: int(p.stem))


def read_rgb(cv2, path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(cv2, path: Path, img) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bool(cv2.imwrite(str(path), bgr))


def resize_if_needed(cv2, img, width: int, height: int):
    h, w = img.shape[:2]
    if width <= 0 or height <= 0 or (w == width and h == height):
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def load_history_frames(cv2, row: dict[str, str], width: int, height: int):
    input_dir = Path(row.get("input_dir", ""))
    pngs = numeric_pngs(input_dir)
    frames = []
    for path in pngs:
        img = read_rgb(cv2, path)
        if img is None:
            continue
        frames.append(resize_if_needed(cv2, img, width, height))
    return frames, pngs


def clean_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
