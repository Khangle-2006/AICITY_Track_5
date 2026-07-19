from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_report(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f) if row.get("status") == "ok"}


def read_video_bgr(cv2, path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames, fps


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def resize_like(cv2, frame, reference):
    if frame.shape[:2] == reference.shape[:2]:
        return frame
    h, w = reference.shape[:2]
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)


def transition_alpha(index: int, transition: int) -> float:
    return min(1.0, (index + 1) / float(transition))


def sigmoid(np, value):
    return 1.0 / (1.0 + np.exp(-value))


def make_motion_edge_mask(cv2, np, prev_bgr, last_bgr, width: int, motion_tau: float, edge_tau: float, blur: float):
    h, w = last_bgr.shape[:2]
    scale = 1.0
    prev_small = prev_bgr
    last_small = last_bgr
    if width > 0 and w > width:
        scale = width / float(w)
        size = (width, max(2, int(round(h * scale))))
        prev_small = cv2.resize(prev_bgr, size, interpolation=cv2.INTER_AREA)
        last_small = cv2.resize(last_bgr, size, interpolation=cv2.INTER_AREA)

    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    last_gray = cv2.cvtColor(last_small, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        last_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2) / max(scale, 1e-6)
    motion = sigmoid(np, (mag - motion_tau) / max(0.1, motion_tau * 0.5))

    edges = cv2.Canny(last_gray, 50, 130).astype(np.float32) / 255.0
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edge = sigmoid(np, (edges - edge_tau) / 0.20)

    mask = np.clip(0.70 * motion + 0.30 * edge, 0.0, 1.0).astype(np.float32)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur, sigmaY=blur)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)[:, :, None]


def parse_variant(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise ValueError(f"variant must be label=strength, got {value!r}")
    label, raw_strength = value.split("=", 1)
    return label.strip(), float(raw_strength)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build motion/edge masked LTX condition transitions.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--transition", type=int, default=16)
    parser.add_argument("--variant", action="append", required=True, help="label=strength")
    parser.add_argument("--motion_width", type=int, default=640)
    parser.add_argument("--motion_tau", type=float, default=1.0)
    parser.add_argument("--edge_tau", type=float, default=0.25)
    parser.add_argument("--mask_blur", type=float, default=3.0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows = read_manifest(args.manifest)
    reports = read_report(args.ltx_report)
    variants = [parse_variant(value) for value in args.variant]
    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    for label, strength in variants:
        out_dir = args.out_root / label
        report_rows = []
        for position, row in enumerate(rows, start=1):
            sample_id = row["sample_id"]
            source = reports.get(sample_id)
            if source is None:
                print(f"{label} [{position}/{len(rows)}] {sample_id} | skipped", flush=True)
                continue
            condition_paths = ((row.get("frames") or {}).get("condition") or [])
            if len(condition_paths) < 2:
                raise RuntimeError(f"{sample_id}: need at least two condition frames")
            prev_rgb = read_rgb(cv2, Path(condition_paths[-2]))
            last_rgb = read_rgb(cv2, Path(condition_paths[-1]))
            if prev_rgb is None or last_rgb is None:
                raise RuntimeError(f"{sample_id}: missing condition frame")
            prev_bgr = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2BGR)
            last_bgr = cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR)

            ltx_frames, fps = read_video_bgr(cv2, Path(source["video"]))
            base = resize_like(cv2, last_bgr, ltx_frames[0])
            prev = resize_like(cv2, prev_bgr, base)
            mask = make_motion_edge_mask(
                cv2,
                np,
                prev,
                base,
                args.motion_width,
                args.motion_tau,
                args.edge_tau,
                args.mask_blur,
            )

            frames = []
            for index, frame in enumerate(ltx_frames):
                alpha = transition_alpha(index, args.transition)
                masked_alpha = np.clip(alpha + strength * mask * (1.0 - alpha), 0.0, 1.0)
                blended = np.clip(
                    (1.0 - masked_alpha) * base.astype(np.float32)
                    + masked_alpha * frame.astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)
                frames.append(blended)

            video_path = out_dir / "videos" / f"{sample_id}.mp4"
            write_video(cv2, frames, video_path, fps)
            report_rows.append(
                {
                    "sample_id": sample_id,
                    "video": str(video_path),
                    "status": "ok",
                    "source_fps": source.get("source_fps", ""),
                    "future_frames": len(frames),
                    "transition_frames": args.transition,
                    "mask_strength": strength,
                    "notes": "motion/edge masked transition alpha",
                }
            )
            print(f"{label} [{position}/{len(rows)}] {sample_id} | ok", flush=True)

        with (out_dir / "report.csv").open("w", encoding="utf-8", newline="") as f:
            fields = list(report_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report_rows)


if __name__ == "__main__":
    main()
