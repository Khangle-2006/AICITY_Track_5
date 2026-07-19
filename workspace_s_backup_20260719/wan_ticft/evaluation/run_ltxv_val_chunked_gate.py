from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.run_ltxv_val_gate import (
    align_history_to_reference,
    compatible_history_length,
    read_history,
    read_history_sequential,
    read_rows,
    short_prompt,
)


def compatible_total_length(minimum: int) -> int:
    return ((minimum - 2) // 8 + 1) * 8 + 1


def target_horizon(row: dict, fallback: int) -> int:
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    target = indices.get("target") or []
    if condition and target:
        last = int(condition[-1])
        return max(1, max(int(index) - last for index in target))
    return fallback


def resize_bgr(cv2, frame, width: int, height: int):
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def write_video(cv2, frames, path: Path, fps: float, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(resize_bgr(cv2, frame, width, height))
    writer.release()


def read_video_bgr(cv2, video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open generated video {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"{video_path}: no decoded frames")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    return frames, fps


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "video",
        "status",
        "history_frames",
        "source_fps",
        "future_frames",
        "chunk_frames",
        "chunks",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LTX validation gate with rolling short chunks.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ltx_repo", type=Path, required=True)
    parser.add_argument("--ltx_python", type=Path, required=True)
    parser.add_argument("--pipeline_config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--sample_ids", nargs="*", default=None)
    parser.add_argument(
        "--prompt_mode",
        choices=[
            "legacy",
            "manifest",
            "behavior",
            "minimal",
            "traffic_detail",
            "geometry_lock",
            "copy_bias",
            "dynamic_geometry",
            "dashcam_native",
        ],
        default="traffic_detail",
    )
    parser.add_argument("--max_history_frames", type=int, default=41)
    parser.add_argument("--chunk_frames", type=int, default=16)
    parser.add_argument("--future_frames", type=int, default=0)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip_bad", action="store_true")
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    rows = read_rows(args.manifest, 0, None, args.view_type, args.sample_ids)
    if len(rows) < args.limit:
        raise RuntimeError(f"found only {len(rows)} rows for view_type={args.view_type}")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    selected_rows = []
    for position, row in enumerate(rows, start=1):
        if len(report) >= args.limit:
            break
        sample_id = row["sample_id"]
        condition = (row.get("frame_indices") or {}).get("condition") or []
        if not condition:
            raise RuntimeError(f"{sample_id}: missing condition indices")
        last_index = int(condition[-1])
        history_count = compatible_history_length(last_index + 1, args.max_history_frames)
        if not history_count:
            message = f"{sample_id}: fewer than 9 preceding frames"
            if args.skip_bad:
                print(f"[{position}/{len(rows)}] skip bad | {message}", flush=True)
                continue
            raise RuntimeError(message)

        video_path = Path(row["video_path"])
        decode_mode = "random_seek"
        try:
            frames, source_fps = read_history(cv2, video_path, last_index, history_count)
        except RuntimeError as exc:
            print(f"{sample_id}: random seek failed ({exc}); retrying sequential decode", flush=True)
            frames, source_fps = read_history_sequential(cv2, video_path, last_index, history_count)
            decode_mode = "sequential"

        condition_paths = (row.get("frames") or {}).get("condition") or []
        reference = read_rgb(cv2, Path(condition_paths[-1])) if condition_paths else None
        frames, orientation, mean_error = align_history_to_reference(cv2, frames, reference)
        print(
            f"{sample_id}: selected history decode={decode_mode}, orientation={orientation}, "
            f"condition mean_absdiff={mean_error:.3f}",
            flush=True,
        )
        if reference is not None and mean_error > 3.0:
            message = f"{sample_id}: extracted history mismatch mean_absdiff={mean_error:.3f}"
            if args.skip_bad:
                print(f"[{position}/{len(rows)}] skip bad | {message}", flush=True)
                continue
            raise RuntimeError(message)

        context = [resize_bgr(cv2, frame, args.width, args.height) for frame in frames]
        wanted_future = args.future_frames or target_horizon(row, args.chunk_frames * 3)
        generated_video = []
        generated_future_count = 0
        chunk_index = 0
        raw_root = args.out_dir / "raw" / sample_id
        while generated_future_count < wanted_future:
            chunk_index += 1
            remaining = wanted_future - generated_future_count
            chunk = min(args.chunk_frames, remaining)
            current_history = compatible_history_length(len(context), args.max_history_frames)
            history_frames = context[-current_history:]
            # LTX returns the conditioning guard at current_history - 1, then
            # generated future frames from current_history onward.
            total_frames = compatible_total_length(current_history + chunk)
            history_video = args.out_dir / "history_videos" / f"{sample_id}_chunk{chunk_index:02d}.mp4"
            write_video(cv2, history_frames, history_video, source_fps, args.width, args.height)

            raw_dir = raw_root / f"chunk{chunk_index:02d}"
            raw_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(args.ltx_python),
                str(args.ltx_repo / "inference.py"),
                "--prompt",
                short_prompt(row, args.prompt_mode),
                "--conditioning_media_paths",
                str(history_video),
                "--conditioning_start_frames",
                "0",
                "--conditioning_strengths",
                "1.0",
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--num_frames",
                str(total_frames),
                "--frame_rate",
                str(max(1, round(source_fps))),
                "--seed",
                str(args.seed + chunk_index - 1),
                "--offload_to_cpu",
                "true",
                "--pipeline_config",
                str(args.pipeline_config),
                "--output_path",
                str(raw_dir),
            ]
            print(
                f"[{position}/{len(rows)}] {sample_id} chunk={chunk_index} "
                f"history={current_history} chunk={chunk} total={total_frames}",
                flush=True,
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(args.ltx_repo)
            env["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
            env["TOKENIZERS_PARALLELISM"] = "false"
            subprocess.run(command, check=True, env=env)

            raw_videos = list(raw_dir.glob("*.mp4"))
            if len(raw_videos) != 1:
                raise RuntimeError(f"{sample_id}: expected one chunk output, found {len(raw_videos)}")
            output_frames, _ = read_video_bgr(cv2, raw_videos[0])
            segment = output_frames[current_history - 1 : current_history + chunk]
            if len(segment) != chunk + 1:
                raise RuntimeError(f"{sample_id}: chunk {chunk_index} decoded {len(segment)}/{chunk + 1}")
            if not generated_video:
                generated_video.extend(segment)
            else:
                generated_video.extend(segment[1:])
            context.extend(segment[1:])
            generated_future_count += chunk

        future_path = args.out_dir / "videos" / f"{sample_id}.mp4"
        write_video(cv2, generated_video, future_path, source_fps, args.width, args.height)
        report.append(
            {
                "sample_id": sample_id,
                "video": str(future_path),
                "status": "ok",
                "history_frames": history_count,
                "source_fps": source_fps,
                "future_frames": len(generated_video),
                "chunk_frames": args.chunk_frames,
                "chunks": chunk_index,
            }
        )
        selected_rows.append(row)

    if len(report) != args.limit:
        raise RuntimeError(f"completed only {len(report)}/{args.limit} requested rows")
    write_report(args.out_dir / "report.csv", report)
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(report), "requested": args.limit, "status": "ok"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
