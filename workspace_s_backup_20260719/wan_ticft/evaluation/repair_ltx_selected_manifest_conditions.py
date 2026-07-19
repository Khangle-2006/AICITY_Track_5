"""Repair selected LTX manifests whose materialized frame paths no longer exist.

`run_ltxv_val_gate` writes per-sample history videos next to the generated base
videos. Some source manifests point at temporary materialized condition PNGs
and target PNGs that may have been deleted. This script extracts condition
frames from those history videos and target frames from the original source
video, preserving the original frame indices used for timestamp alignment.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_frames(video_path: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open history video: {video_path}")

    paths: list[str] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_path = out_dir / f"condition_{index:03d}.png"
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"cannot write {frame_path}")
        paths.append(str(frame_path))
        index += 1
    capture.release()

    if not paths:
        raise RuntimeError(f"history video has no readable frames: {video_path}")
    return paths


def extract_indexed_frames(video_path: Path, indices: list[int], out_dir: Path, prefix: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0 and indices and max(int(index) for index in indices) >= frame_count:
        capture.release()
        raise RuntimeError(
            f"target frame index {max(int(index) for index in indices)} exceeds "
            f"{video_path} frame_count={frame_count}"
        )

    paths: list[str] = []
    for output_index, frame_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"cannot read {video_path} at frame {frame_index}")
        frame_path = out_dir / f"{prefix}_{output_index:03d}.png"
        if not cv2.imwrite(str(frame_path), frame):
            capture.release()
            raise RuntimeError(f"cannot write {frame_path}")
        paths.append(str(frame_path))
    capture.release()

    if len(paths) != len(indices):
        raise RuntimeError(f"extracted {len(paths)}/{len(indices)} frames from {video_path}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history_dir", type=Path, required=True)
    parser.add_argument("--frames_root", type=Path, required=True)
    parser.add_argument("--out_manifest", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--skip_target",
        action="store_true",
        help="Only repair condition frames; by default target frames are also materialized from video_path.",
    )
    parser.add_argument(
        "--skip_bad",
        action="store_true",
        help="Skip rows whose history or target frames cannot be read instead of failing.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if args.clean and args.frames_root.exists():
        shutil.rmtree(args.frames_root)
    args.frames_root.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)

    repaired: list[dict] = []
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        try:
            history_video = args.history_dir / f"{sample_id}.mp4"
            if not history_video.exists():
                raise RuntimeError(f"missing history video for {sample_id}: {history_video}")

            condition_dir = args.frames_root / sample_id / "condition"
            if args.clean and condition_dir.exists():
                shutil.rmtree(condition_dir)
            condition_paths = extract_frames(history_video, condition_dir)

            row = dict(row)
            frames = dict(row.get("frames") or {})
            frames["condition"] = condition_paths
            if not args.skip_target:
                target_indices = list((row.get("frame_indices") or {}).get("target") or [])
                if not target_indices:
                    raise RuntimeError(f"{sample_id}: missing frame_indices.target")
                source_video = Path(row.get("video_path") or "")
                target_dir = args.frames_root / sample_id / "target"
                if args.clean and target_dir.exists():
                    shutil.rmtree(target_dir)
                frames["target"] = extract_indexed_frames(source_video, target_indices, target_dir, "target")
            row["frames"] = frames
            repaired.append(row)
            print(
                f"[{position}/{len(rows)}] {sample_id}: "
                f"condition_frames={len(condition_paths)} target_frames={len(frames.get('target') or [])}",
                flush=True,
            )
        except Exception as exc:
            if not args.skip_bad:
                raise
            print(f"[{position}/{len(rows)}] {sample_id}: skip_bad {exc}", flush=True)

    if not repaired:
        raise RuntimeError("no repaired rows")
    args.out_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in repaired),
        encoding="utf-8",
    )
    print(f"output={args.out_manifest}")


if __name__ == "__main__":
    main()
