from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2

from baseline.frame_io import import_cv2_np, numeric_pngs
from baseline.verify import verify_sample_output
from wan_ticft.evaluation.run_ltxv_test_bdd import short_prompt


REPORT_FIELDS = [
    "sample_id",
    "output_name",
    "view_type",
    "phase",
    "history_frames",
    "target_N",
    "output_frames",
    "chunk_frames",
    "chunks",
    "status",
    "notes",
]


def read_rows(path: Path, offset: int, limit: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("view_type") == "bdd"]
    return rows[offset : offset + limit if limit else None]


def compatible_history_length(frame_count: int, maximum: int) -> int:
    frame_count = min(frame_count, maximum)
    return ((frame_count - 1) // 8) * 8 + 1 if frame_count >= 9 else 0


def compatible_total_length(minimum: int) -> int:
    return int(math.ceil((minimum - 1) / 8) * 8 + 1)


def resize_bgr(frame, width: int, height: int):
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def load_history(frame_paths: list[Path], width: int, height: int):
    frames = []
    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot read {path}")
        frames.append(resize_bgr(frame, width, height))
    return frames


def write_video(frames, path: Path, fps: float, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(resize_bgr(frame, width, height))
    writer.release()


def read_video_bgr(video_path: Path):
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
    return frames


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REPORT_FIELDS})
    temp.replace(path)


def write_submission_frames(frames, out_dir: Path, start_index: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for offset, frame in enumerate(frames):
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
        path = out_dir / f"{start_index + offset}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"cannot write {path}")
    return start_index + len(frames)


def gpu_free_mib(gpu_id: str) -> int | None:
    if not gpu_id:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                gpu_id,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    try:
        return int(result.stdout.strip().splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


def wait_for_gpu(gpu_id: str, minimum_mib: int, poll_seconds: int) -> None:
    if not gpu_id or minimum_mib <= 0:
        return
    while True:
        free = gpu_free_mib(gpu_id)
        if free is None or free >= minimum_mib:
            if free is not None:
                print(f"GPU {gpu_id}: free={free} MiB", flush=True)
            return
        print(f"GPU {gpu_id}: free={free} MiB < {minimum_mib} MiB; waiting {poll_seconds}s", flush=True)
        time.sleep(poll_seconds)


def run_with_retries(command, env, retries: int, retry_sleep: int) -> None:
    attempt = 0
    while True:
        attempt += 1
        try:
            subprocess.run(command, check=True, env=env)
            return
        except subprocess.CalledProcessError:
            if attempt > retries:
                raise
            print(f"chunk command failed; retrying in {retry_sleep}s ({attempt}/{retries})", flush=True)
            time.sleep(retry_sleep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BDD test frames with rolling chunked LTX-Video.")
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ltx_repo", type=Path, required=True)
    parser.add_argument("--ltx_python", type=Path, required=True)
    parser.add_argument("--pipeline_config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_history", type=int, default=41)
    parser.add_argument("--chunk_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--prompt_mode",
        choices=["legacy", "minimal", "traffic_detail", "geometry_lock", "copy_bias", "dynamic_geometry"],
        default="traffic_detail",
    )
    parser.add_argument("--gpu_id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0])
    parser.add_argument("--min_gpu_free_mib", type=int, default=12000)
    parser.add_argument("--gpu_poll_seconds", type=int, default=30)
    parser.add_argument("--chunk_retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cv2_module, _ = import_cv2_np()
    rows = read_rows(args.test_csv, args.offset, args.limit)
    if not rows:
        raise RuntimeError("no BDD test rows selected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "report.csv"
    report: list[dict[str, object]] = []
    if args.resume and report_path.exists():
        with report_path.open(encoding="utf-8", newline="") as handle:
            report = list(csv.DictReader(handle))

    completed = {str(row.get("output_name")) for row in report if row.get("status") == "ok"}
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        output_dir = args.out_dir / output_name
        existing = verify_sample_output(output_dir, target_n)
        if args.resume and (existing.ok or output_name in completed):
            if not existing.ok:
                raise RuntimeError(f"{output_name}: report says complete but files are bad: {existing.notes}")
            if output_name not in completed:
                report.append(
                    {
                        "sample_id": sample_id,
                        "output_name": output_name,
                        "view_type": "bdd",
                        "phase": row.get("phase", ""),
                        "history_frames": "",
                        "target_N": target_n,
                        "output_frames": existing.frames_found,
                        "chunk_frames": args.chunk_frames,
                        "chunks": "",
                        "status": "ok",
                        "notes": "resume existing",
                    }
                )
                write_report(report_path, report)
            print(f"[{position}/{len(rows)}] {output_name} | resume existing", flush=True)
            continue

        history_paths = numeric_pngs(Path(row["input_dir"]))
        history_count = compatible_history_length(len(history_paths), args.max_history)
        if not history_count:
            raise RuntimeError(f"{sample_id}: fewer than 9 readable history PNGs")
        history_paths = history_paths[-history_count:]
        context = load_history(history_paths, args.width, args.height)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        chunk_index = 0
        while written < target_n:
            chunk_index += 1
            chunk = min(args.chunk_frames, target_n - written)
            current_history = compatible_history_length(len(context), args.max_history)
            history_frames = context[-current_history:]
            total_frames = compatible_total_length(current_history + chunk)
            history_video = args.out_dir / "_history" / f"{output_name}_chunk{chunk_index:02d}.mp4"
            write_video(history_frames, history_video, args.fps, args.width, args.height)

            raw_dir = args.out_dir / "_raw" / output_name / f"chunk{chunk_index:02d}"
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            raw_dir.mkdir(parents=True)
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
                str(max(1, round(args.fps))),
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
                f"[{position}/{len(rows)}] {output_name} chunk={chunk_index} "
                f"history={current_history} chunk={chunk} total={total_frames}",
                flush=True,
            )
            wait_for_gpu(args.gpu_id, args.min_gpu_free_mib, args.gpu_poll_seconds)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(args.ltx_repo)
            env["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
            env["TOKENIZERS_PARALLELISM"] = "false"
            run_with_retries(command, env, args.chunk_retries, args.retry_sleep)

            raw_videos = list(raw_dir.glob("*.mp4"))
            if len(raw_videos) != 1:
                raise RuntimeError(f"{sample_id}: expected one chunk output, found {len(raw_videos)}")
            output_frames = read_video_bgr(raw_videos[0])
            segment = output_frames[current_history - 1 : current_history + chunk]
            if len(segment) != chunk + 1:
                raise RuntimeError(f"{sample_id}: chunk {chunk_index} decoded {len(segment)}/{chunk + 1}")
            future = segment[1:]
            written = write_submission_frames(future, output_dir, written)
            context.extend(future)

        verified = verify_sample_output(output_dir, target_n)
        if not verified.ok:
            raise RuntimeError(f"{sample_id}: output verification failed: {verified.notes}")
        result = {
            "sample_id": sample_id,
            "output_name": output_name,
            "view_type": "bdd",
            "phase": row.get("phase", ""),
            "history_frames": history_count,
            "target_N": target_n,
            "output_frames": verified.frames_found,
            "chunk_frames": args.chunk_frames,
            "chunks": chunk_index,
            "status": "ok",
            "notes": "",
        }
        report = [item for item in report if item.get("output_name") != output_name]
        report.append(result)
        write_report(report_path, report)

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"rows": len(report), "status": "ok", "report": str(report_path)}, handle, indent=2)
    print(json.dumps({"rows": len(report), "status": "ok", "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
