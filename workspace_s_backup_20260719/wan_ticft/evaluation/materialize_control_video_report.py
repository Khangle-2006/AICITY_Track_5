"""Convert a video report into IC-LoRA control videos.

The output intentionally removes RGB texture and keeps structural cues:
- R: Canny edges for layout and object boundaries.
- G: optical-flow magnitude for temporal motion.
- B: photometric residual after backward warping, a cheap occlusion/error cue.

This is meant as a reference/control signal for LTX IC-LoRA, not a final
postprocess. It follows the control-video idea used by Canny/flow/depth video
diffusion papers while staying lightweight enough for BDD gates.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_report(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok" and row.get("video")]


def read_video(path: Path, limit: int = 0) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    while not limit or len(frames) < limit:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames, fps


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def normalize_channel(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(values.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def control_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    prev_gray: np.ndarray | None = None
    out: list[np.ndarray] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        if prev_gray is None:
            flow_mag = np.zeros_like(gray)
            residual = np.zeros_like(gray)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag = np.sqrt(flow[..., 0] * flow[..., 0] + flow[..., 1] * flow[..., 1])
            flow_mag = normalize_channel(mag, 16.0)

            h, w = gray.shape
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            # Backward sample previous frame at current minus forward flow.
            warped_prev = cv2.remap(
                prev_gray,
                xs - flow[..., 0],
                ys - flow[..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            residual = normalize_channel(np.abs(gray.astype(np.float32) - warped_prev.astype(np.float32)), 3.0)
            residual = cv2.GaussianBlur(residual, (0, 0), 1.0)

        # BGR write: B=residual, G=flow, R=edge. When loaded as RGB by trainer,
        # this becomes R=edge, G=flow, B=residual.
        out.append(cv2.merge([residual, flow_mag, edges]))
        prev_gray = gray
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.out_dir / "videos"

    rows = read_report(args.input_report)
    if args.limit:
        rows = rows[: args.limit]
    output_rows = []
    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        try:
            frames, fps = read_video(Path(row["video"]), args.frames)
            controls = control_frames(frames)
            out_path = videos_dir / f"{sample_id}.mp4"
            write_video(out_path, controls, fps)
            status = "ok"
            notes = ""
        except Exception as exc:  # noqa: BLE001
            out_path = Path("")
            status = "bad"
            notes = repr(exc)
            controls = []
            fps = 0.0
        output_rows.append(
            {
                "sample_id": sample_id,
                "video": str(out_path),
                "status": status,
                "frames": len(controls),
                "fps": f"{fps:.8f}" if fps else "",
                "notes": notes,
            }
        )
        print(f"[{index}/{len(rows)}] {sample_id} | {status} {notes}", flush=True)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "frames", "fps", "notes"])
        writer.writeheader()
        writer.writerows(output_rows)
    bad = [row for row in output_rows if row["status"] != "ok"]
    print(f"report={report_path}")
    print(f"bad={len(bad)}/{len(output_rows)}")
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
