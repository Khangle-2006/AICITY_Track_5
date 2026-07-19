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


def parse_variant(value: str) -> tuple[str, float, float]:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("variant must be label:alpha:diff_threshold")
    label, raw_alpha, raw_thr = parts
    alpha = float(raw_alpha)
    threshold = float(raw_thr)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(value)
    return label, alpha, threshold


def denoise_sequence(cv2, np, frames, alpha: float, threshold: float):
    out = [frames[0]]
    state = frames[0].astype(np.float32)
    for frame in frames[1:]:
        cur = frame.astype(np.float32)
        diff = np.mean(np.abs(cur - state), axis=2) / 255.0
        # High difference likely means true motion/edge; leave it mostly alone.
        weight = alpha * np.exp(-diff / max(1e-6, threshold))
        weight = cv2.GaussianBlur(weight.astype(np.float32), (0, 0), sigmaX=1.5, sigmaY=1.5)
        weight = np.clip(weight, 0.0, alpha)[..., None]
        filtered = (1.0 - weight) * cur + weight * state
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)
        out.append(filtered)
        # Update state slowly from filtered frame to avoid one-frame noise accumulation.
        state = 0.75 * filtered.astype(np.float32) + 0.25 * cur
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal denoise a generated video report without geometric warping.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="label:alpha:diff_threshold")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    report = read_report(args.report)
    variants = [parse_variant(item) for item in args.variant]
    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    for label, alpha, threshold in variants:
        rows = []
        for pos, (sample_id, row) in enumerate(sorted(report.items()), start=1):
            frames, fps = read_video(cv2, Path(row["video"]))
            out_frames = denoise_sequence(cv2, np, frames, alpha, threshold)
            out_path = args.out_root / label / "videos" / f"{sample_id}.mp4"
            write_video(cv2, out_frames, out_path, fps)
            rows.append(
                {
                    "sample_id": sample_id,
                    "video": str(out_path),
                    "status": "ok",
                    "source_fps": f"{fps:.6f}",
                    "future_frames": len(out_frames),
                    "alpha": f"{alpha:.6f}",
                    "threshold": f"{threshold:.6f}",
                }
            )
            print(f"{label} [{pos}/{len(report)}] {sample_id} | ok", flush=True)
        report_path = args.out_root / label / "report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
