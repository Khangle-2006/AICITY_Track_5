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


def parse_variant(value: str) -> tuple[str, int, float, float, float]:
    parts = value.split(":")
    if len(parts) != 5:
        raise ValueError("variant must be label:odd_kernel:alpha:motion_thr:edge_thr")
    label, raw_kernel, raw_alpha, raw_motion, raw_edge = parts
    kernel = int(raw_kernel)
    alpha = float(raw_alpha)
    motion_thr = float(raw_motion)
    edge_thr = float(raw_edge)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError(f"kernel must be a positive odd integer: {value!r}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha out of range: {value!r}")
    return label, kernel, alpha, motion_thr, edge_thr


def protection_mask(cv2, np, prev, cur, motion_thr: float, edge_thr: float):
    gray = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    motion = cv2.absdiff(gray, prev_gray).astype(np.float32) / 255.0
    motion = np.clip((motion - motion_thr) / max(1e-6, 0.20 - motion_thr), 0.0, 1.0)

    edges = cv2.Canny(gray, int(edge_thr), int(edge_thr * 2.5)).astype(np.float32) / 255.0
    kernel = np.ones((5, 5), np.uint8)
    motion = cv2.dilate(motion, kernel, iterations=2)
    edges = cv2.dilate(edges, kernel, iterations=2)
    protect = np.maximum(motion, edges)
    protect = cv2.GaussianBlur(protect, (21, 21), 0)
    return np.clip(protect, 0.0, 1.0)[..., None]


def compose(cv2, np, prev_base, base, cosmos, kernel: int, alpha: float, motion_thr: float, edge_thr: float):
    if cosmos.shape[:2] != base.shape[:2]:
        cosmos = cv2.resize(cosmos, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
    base_f = base.astype(np.float32)
    cosmos_f = cosmos.astype(np.float32)
    base_low = cv2.GaussianBlur(base_f, (kernel, kernel), 0)
    cosmos_low = cv2.GaussianBlur(cosmos_f, (kernel, kernel), 0)
    bg_weight = 1.0 - protection_mask(cv2, np, prev_base, base, motion_thr, edge_thr)
    out = base_f + alpha * bg_weight * (cosmos_low - base_low)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Masked Cosmos3 background prior over a base video report.")
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--cosmos_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="label:odd_kernel:alpha:motion_thr:edge_thr")
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

    for label, kernel, alpha, motion_thr, edge_thr in variants:
        rows = []
        for pos, sample_id in enumerate(common, start=1):
            base_frames, base_fps = read_video(cv2, Path(base_report[sample_id]["video"]))
            cosmos_frames, cosmos_fps = read_video(cv2, Path(cosmos_report[sample_id]["video"]))
            fps = base_fps or cosmos_fps or 30.0
            out_frames = []
            for index, base in enumerate(base_frames):
                prev_base = base_frames[max(0, index - 1)]
                cosmos_index = int(round(index * cosmos_fps / fps)) if fps > 0 and cosmos_fps > 0 else index
                cosmos_index = min(max(0, cosmos_index), len(cosmos_frames) - 1)
                out_frames.append(
                    compose(
                        cv2,
                        np,
                        prev_base,
                        base,
                        cosmos_frames[cosmos_index],
                        kernel,
                        alpha,
                        motion_thr,
                        edge_thr,
                    )
                )
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
                    "motion_thr": f"{motion_thr:.6f}",
                    "edge_thr": f"{edge_thr:.6f}",
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
