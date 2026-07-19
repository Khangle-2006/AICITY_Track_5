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


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def transition_alpha(index: int, transition: int, curve: str) -> float:
    t = min(1.0, (index + 1) / float(transition))
    if curve == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    return t


def apply_fixed_color_anchor(np, frames, base, transition: int, strength: float):
    if strength <= 0.0:
        return frames
    reference = frames[min(transition - 1, len(frames) - 1)].astype(np.float32)
    base_f = base.astype(np.float32)
    ref_mean = reference.mean(axis=(0, 1))
    ref_std = reference.std(axis=(0, 1)).clip(1.0)
    base_mean = base_f.mean(axis=(0, 1))
    base_std = base_f.std(axis=(0, 1)).clip(1.0)
    gain = np.clip(base_std / ref_std, 0.9, 1.1)
    bias = np.clip(base_mean - ref_mean * gain, -12.0, 12.0)
    anchored = []
    for frame in frames:
        corrected = frame.astype(np.float32) * gain + bias
        anchored.append(
            np.clip(
                (1.0 - strength) * frame.astype(np.float32) + strength * corrected,
                0,
                255,
            ).astype(np.uint8)
        )
    return anchored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--transitions", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--curve", choices=["linear", "smoothstep"], default="linear")
    parser.add_argument("--color_anchor_strength", type=float, default=0.0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows = read_manifest(args.manifest)
    reports = read_report(args.ltx_report)
    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    for transition in args.transitions:
        label = f"ltx_transition_{transition}"
        if args.curve != "linear":
            label += f"_{args.curve}"
        if args.color_anchor_strength > 0.0:
            label += f"_color{round(args.color_anchor_strength * 100):02d}"
        output_dir = args.out_root / label
        report_rows = []
        for position, row in enumerate(rows, start=1):
            sample_id = row["sample_id"]
            source = reports.get(sample_id)
            if source is None:
                continue
            condition_paths = ((row.get("frames") or {}).get("condition") or [])
            condition_rgb = read_rgb(cv2, Path(condition_paths[-1])) if condition_paths else None
            if condition_rgb is None:
                raise RuntimeError(f"{sample_id}: missing last condition frame")
            condition = cv2.cvtColor(condition_rgb, cv2.COLOR_RGB2BGR)

            cap = cv2.VideoCapture(source["video"])
            if not cap.isOpened():
                raise RuntimeError(f"{sample_id}: cannot open {source['video']}")
            fps = float(cap.get(cv2.CAP_PROP_FPS) or source.get("source_fps") or 30.0)
            source_frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                source_frames.append(frame)
            cap.release()
            if not source_frames:
                raise RuntimeError(f"{sample_id}: no decoded LTX frames")
            if condition.shape[:2] != source_frames[0].shape[:2]:
                base = cv2.resize(
                    condition,
                    (source_frames[0].shape[1], source_frames[0].shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                base = condition
            source_frames = apply_fixed_color_anchor(
                np, source_frames, base, transition, args.color_anchor_strength
            )
            frames = []
            for index, frame in enumerate(source_frames):
                alpha = transition_alpha(index, transition, args.curve)
                blended = np.clip(
                    (1.0 - alpha) * base.astype(np.float32)
                    + alpha * frame.astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)
                frames.append(blended)
            video_path = output_dir / "videos" / f"{sample_id}.mp4"
            write_video(cv2, frames, video_path, fps)
            report_rows.append(
                {
                    "sample_id": sample_id,
                    "video": str(video_path),
                    "status": "ok",
                    "source_fps": source.get("source_fps", ""),
                    "future_frames": len(frames),
                    "transition_frames": transition,
                    "transition_curve": args.curve,
                    "color_anchor_strength": args.color_anchor_strength,
                }
            )
            print(f"{label} [{position}/{len(rows)}] {sample_id}", flush=True)

        with (output_dir / "report.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=report_rows[0].keys())
            writer.writeheader()
            writer.writerows(report_rows)


if __name__ == "__main__":
    main()
