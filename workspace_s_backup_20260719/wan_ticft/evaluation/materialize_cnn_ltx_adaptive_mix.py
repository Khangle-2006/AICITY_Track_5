from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np


def read_report(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("status") == "ok"}


def read_video(cv2, path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames, fps


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def parse_variant(value: str) -> tuple[str, str, float, float, float]:
    parts = value.split(":")
    if len(parts) != 5:
        raise ValueError("variant must be label:mode:min_alpha:max_alpha:transition")
    label, mode, raw_min, raw_max, raw_transition = parts
    if mode not in {"ramp", "motion", "edge"}:
        raise ValueError(f"unknown mode {mode!r}")
    min_alpha = float(raw_min)
    max_alpha = float(raw_max)
    transition = float(raw_transition)
    if not 0 <= min_alpha <= max_alpha <= 1:
        raise ValueError(value)
    return label, mode, min_alpha, max_alpha, transition


def alpha_map(cv2, np, cnn, prev_cnn, mode: str, min_alpha: float, max_alpha: float, t: float):
    if mode == "ramp":
        return min_alpha + (max_alpha - min_alpha) * t
    gray = cv2.cvtColor(cnn, cv2.COLOR_RGB2GRAY)
    if mode == "motion":
        prev_gray = cv2.cvtColor(prev_cnn, cv2.COLOR_RGB2GRAY)
        motion = cv2.absdiff(gray, prev_gray).astype(np.float32) / 255.0
        protect = np.clip((motion - 0.025) / 0.12, 0.0, 1.0)
        protect = cv2.dilate(protect, np.ones((7, 7), np.uint8), iterations=1)
        protect = cv2.GaussianBlur(protect, (21, 21), 0)
    else:
        edges = cv2.Canny(gray, 45, 120).astype(np.float32) / 255.0
        protect = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        protect = cv2.GaussianBlur(protect, (21, 21), 0)
    # LTX helps distribution, CNN helps sharp moving/high-edge regions.
    alpha = min_alpha + (max_alpha - min_alpha) * t
    return (alpha * (1.0 - 0.75 * protect))[..., None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive CNN/LTX video report mixer.")
    parser.add_argument("--cnn_report", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="label:mode:min_alpha:max_alpha:transition")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    cnn_report = read_report(args.cnn_report)
    ltx_report = read_report(args.ltx_report)
    variants = [parse_variant(item) for item in args.variant]
    common = sorted(set(cnn_report) & set(ltx_report))
    if not common:
        raise RuntimeError("reports have no common sample_id")
    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    for label, mode, min_alpha, max_alpha, transition in variants:
        rows = []
        for pos, sample_id in enumerate(common, start=1):
            cnn_frames, fps = read_video(cv2, Path(cnn_report[sample_id]["video"]))
            ltx_frames, ltx_fps = read_video(cv2, Path(ltx_report[sample_id]["video"]))
            count = min(len(cnn_frames), len(ltx_frames))
            out_frames = []
            for index in range(count):
                cnn = cnn_frames[index]
                ltx = ltx_frames[index]
                if ltx.shape[:2] != cnn.shape[:2]:
                    ltx = cv2.resize(ltx, (cnn.shape[1], cnn.shape[0]), interpolation=cv2.INTER_CUBIC)
                t = min(1.0, (index + 1) / max(1.0, transition))
                prev = cnn_frames[max(0, index - 1)]
                alpha = alpha_map(cv2, np, cnn, prev, mode, min_alpha, max_alpha, t)
                mixed = np.clip(
                    (1.0 - alpha) * cnn.astype(np.float32) + alpha * ltx.astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)
                out_frames.append(mixed)
            out_path = args.out_root / label / "videos" / f"{sample_id}.mp4"
            write_video(cv2, out_frames, out_path, fps or ltx_fps)
            rows.append(
                {
                    "sample_id": sample_id,
                    "video": str(out_path),
                    "status": "ok",
                    "source_fps": f"{fps or ltx_fps:.6f}",
                    "future_frames": len(out_frames),
                    "mode": mode,
                    "min_alpha": f"{min_alpha:.6f}",
                    "max_alpha": f"{max_alpha:.6f}",
                    "transition": f"{transition:.6f}",
                }
            )
            print(f"{label} [{pos}/{len(common)}] {sample_id} | ok", flush=True)
        report_path = args.out_root / label / "report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
