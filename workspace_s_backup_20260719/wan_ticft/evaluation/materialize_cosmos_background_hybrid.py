from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np


def read_report(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row.get("status") == "ok" and row.get("video")
        }


def read_video(cv2, path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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


def parse_variant(value: str) -> tuple[str, int, float]:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"variant must be label:sigma:alpha, got {value!r}")
    label, raw_sigma, raw_alpha = parts
    sigma = int(raw_sigma)
    alpha = float(raw_alpha)
    if sigma <= 0 or sigma % 2 == 0:
        raise ValueError(f"sigma/kernel must be a positive odd integer: {value!r}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha out of range: {value!r}")
    return label, sigma, alpha


def lowfreq_delta(cv2, np, base, cosmos, kernel: int, alpha: float):
    if cosmos.shape[:2] != base.shape[:2]:
        cosmos = cv2.resize(cosmos, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
    base_f = base.astype(np.float32)
    cosmos_f = cosmos.astype(np.float32)
    base_low = cv2.GaussianBlur(base_f, (kernel, kernel), 0)
    cosmos_low = cv2.GaussianBlur(cosmos_f, (kernel, kernel), 0)
    return np.clip(base_f + alpha * (cosmos_low - base_low), 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject Cosmos3 low-frequency background into a base video report.")
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--cosmos_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="label:odd_kernel:alpha")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    base_report = read_report(args.base_report)
    cosmos_report = read_report(args.cosmos_report)
    variants = [parse_variant(item) for item in args.variant]

    common = sorted(set(base_report) & set(cosmos_report))
    if not common:
        raise RuntimeError("reports have no common sample_id")

    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    for label, kernel, alpha in variants:
        rows = []
        for pos, sample_id in enumerate(common, start=1):
            base_frames, base_fps = read_video(cv2, Path(base_report[sample_id]["video"]))
            cosmos_frames, cosmos_fps = read_video(cv2, Path(cosmos_report[sample_id]["video"]))
            fps = base_fps or cosmos_fps or 30.0
            out_frames = []
            for index, base in enumerate(base_frames):
                # Map by time, not raw index, because Cosmos3 anchor currently writes 24 fps.
                cosmos_index = int(round(index * cosmos_fps / fps)) if fps > 0 and cosmos_fps > 0 else index
                cosmos_index = min(max(0, cosmos_index), len(cosmos_frames) - 1)
                out_frames.append(lowfreq_delta(cv2, np, base, cosmos_frames[cosmos_index], kernel, alpha))
            out_path = args.out_root / label / "videos" / f"{sample_id}.mp4"
            write_video(cv2, out_frames, out_path, fps)
            rows.append(
                {
                    "sample_id": sample_id,
                    "video": str(out_path),
                    "status": "ok",
                    "source_fps": f"{fps:.6f}",
                    "future_frames": len(out_frames),
                    "cosmos_fps": f"{cosmos_fps:.6f}",
                    "kernel": kernel,
                    "alpha": f"{alpha:.6f}",
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
