"""Apply chunk detail donor to BDD test folders from an existing submission."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


def numeric_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"), key=lambda p: int(p.stem))


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def blur(frame: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(frame.astype(np.float32), (0, 0), sigma)


def lap_strength(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    return cv2.GaussianBlur(lap, (0, 0), 1.0)


def restore_base_lowfreq(out: np.ndarray, base: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    return np.clip(out.astype(np.float32) - blur(out, sigma) + blur(base, sigma), 0, 255).astype(np.uint8)


def detail_donor(
    base: np.ndarray,
    donor: np.ndarray,
    alpha: float,
    sigma: float,
    threshold: float,
    require_sharper: bool,
    restore_luma: bool,
) -> np.ndarray:
    if donor.shape[:2] != base.shape[:2]:
        donor = cv2.resize(donor, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
    b = base.astype(np.float32)
    d = donor.astype(np.float32)
    detail = d - blur(d, sigma)
    diff = np.mean(np.abs(blur(b, sigma * 1.4) - blur(d, sigma * 1.4)), axis=2)
    mask = np.clip((threshold - diff) / threshold, 0.0, 1.0)
    if require_sharper:
        sharp_delta = lap_strength(donor) - lap_strength(base)
        mask *= np.clip((sharp_delta + 2.0) / 10.0, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2)[..., None]
    out = np.clip(b + alpha * detail * mask, 0, 255).astype(np.uint8)
    return restore_base_lowfreq(out, base) if restore_luma else out


def write_rgb(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"cannot write {path}")


def read_bdd_rows(test_csv: Path) -> list[dict[str, str]]:
    with test_csv.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("view_type") == "bdd"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--donor_dir", type=Path, required=True)
    parser.add_argument("--out_detail", type=Path, required=True)
    parser.add_argument("--out_strict", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    for out_dir in (args.out_detail, args.out_strict):
        if args.clean and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_bdd_rows(args.test_csv)
    for idx, row in enumerate(rows, 1):
        sample = row["source_folder"]
        expected = int(row["target_N"])
        if len(numeric_pngs(args.out_detail / sample)) == expected and len(numeric_pngs(args.out_strict / sample)) == expected:
            print(f"[{idx}/{len(rows)}] {sample} resume existing", flush=True)
            continue
        base_paths = numeric_pngs(args.base_dir / sample)
        donor_paths = numeric_pngs(args.donor_dir / sample)
        if len(base_paths) != expected or len(donor_paths) != expected:
            raise RuntimeError(f"{sample}: expected={expected} base={len(base_paths)} donor={len(donor_paths)}")
        for frame_idx, (base_path, donor_path) in enumerate(zip(base_paths, donor_paths)):
            base = read_rgb(base_path)
            donor = read_rgb(donor_path)
            detail = detail_donor(base, donor, alpha=0.55, sigma=1.15, threshold=18.0, require_sharper=False, restore_luma=False)
            strict = detail_donor(base, donor, alpha=0.55, sigma=1.15, threshold=10.0, require_sharper=True, restore_luma=True)
            write_rgb(args.out_detail / sample / f"{frame_idx}.png", detail)
            write_rgb(args.out_strict / sample / f"{frame_idx}.png", strict)
        print(f"[{idx}/{len(rows)}] {sample} frames={expected}", flush=True)


if __name__ == "__main__":
    main()
