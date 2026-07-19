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
    write_video,
)


REPORT_FIELDS = [
    "sample_id",
    "dataset_part",
    "view_type",
    "phase",
    "video",
    "status",
    "notes",
]


def numeric_png_name(index: int) -> str:
    return f"{index}.png"


def sample_slug(sample_id: str) -> str:
    parts = sample_id.split("_")
    for part in parts:
        if part.startswith("video"):
            return part
    return sample_id


def requested_future_frames(row: dict, override: int) -> int:
    if override > 0:
        return override
    targets = (row.get("frames") or {}).get("target") or []
    if targets:
        return len(targets)
    for key in ("target_N", "target_frames"):
        value = row.get(key)
        if value not in (None, ""):
            return int(value)
    ticft = row.get("ticft") or {}
    if ticft.get("target_frames"):
        return int(ticft["target_frames"])
    raise RuntimeError(f"{row.get('sample_id', '')}: cannot infer future frame count")


def read_history_from_input_dir(cv2, input_dir: Path, last_index: int, count: int):
    start = last_index - count + 1
    frames = []
    for index in range(start, last_index + 1):
        path = input_dir / f"{index}.png"
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"cannot read {path}")
        frames.append(frame)
    return frames, 30.0


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REPORT_FIELDS})


def run_one(cv2, args, row: dict, seed: int) -> dict:
    sample_id = str(row["sample_id"])
    slug = str(row.get("source_folder") or sample_slug(sample_id))
    condition = (row.get("frame_indices") or {}).get("condition") or []
    last_index = int(condition[-1])
    history_count = compatible_history_length(last_index + 1, args.max_history_frames)
    if not history_count:
        raise RuntimeError(f"{sample_id}: not enough history")
    future_count = requested_future_frames(row, args.future_frames)

    raw_video_path = str(row.get("video_path") or "")
    if raw_video_path:
        video_path = Path(raw_video_path)
        try:
            history, source_fps = read_history(cv2, video_path, last_index, history_count)
            decode_mode = "random_seek"
        except RuntimeError:
            history, source_fps = read_history_sequential(cv2, video_path, last_index, history_count)
            decode_mode = "sequential"
    else:
        history, source_fps = read_history_from_input_dir(cv2, Path(row["input_dir"]), last_index, history_count)
        source_fps = float((row.get("frame_indices") or {}).get("fps") or source_fps)
        decode_mode = "input_dir"
    condition_paths = (row.get("frames") or {}).get("condition") or []
    reference = read_rgb(cv2, Path(condition_paths[-1])) if condition_paths else None
    history, orientation, mean_error = align_history_to_reference(cv2, history, reference)
    print(
        f"{sample_id} seed={seed} history={history_count} decode={decode_mode}/{orientation} "
        f"diff={mean_error:.3f} fps={source_fps:.3f}",
        flush=True,
    )

    label = f"seed{seed}"
    input_path = args.out_dir / "_inputs" / f"{sample_id}_history.mp4"
    raw_dir = args.out_dir / "_raw" / label / sample_id
    frame_root = args.out_dir / label / "frames" / slug
    video_path = args.out_dir / label / "videos" / f"{sample_id}.mp4"
    if args.resume and video_path.exists() and frame_root.exists():
        existing = len(list(frame_root.glob("*.png")))
        if existing >= future_count:
            return {
                "sample_id": sample_id,
                "dataset_part": row.get("dataset_part", ""),
                "view_type": row.get("view_type", ""),
                "phase": row.get("phase", ""),
                "video": str(video_path),
                "status": "ok",
                "notes": "resume",
            }

    write_video(cv2, history, input_path, source_fps)
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    if frame_root.exists():
        shutil.rmtree(frame_root)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.ltx_python),
        str(args.ltx_repo / "inference.py"),
        "--prompt",
        short_prompt(row, args.prompt_mode),
        "--conditioning_media_paths",
        str(input_path),
        "--conditioning_start_frames",
        "0",
        "--conditioning_strengths",
        "1.0",
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_frames",
        str(args.num_frames),
        "--frame_rate",
        str(max(1, round(source_fps))),
        "--seed",
        str(seed),
        "--image_cond_noise_scale",
        f"{args.image_cond_noise_scale:.4f}",
        "--offload_to_cpu",
        "true",
        "--pipeline_config",
        str(args.pipeline_config),
        "--output_path",
        str(raw_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.ltx_repo)
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["LTX_VAE_TILING"] = args.vae_tiling
    subprocess.run(command, check=True, env=env)

    raw_videos = sorted(raw_dir.glob("*.mp4"))
    if len(raw_videos) != 1:
        raise RuntimeError(f"{sample_id} seed={seed}: expected one raw mp4, found {len(raw_videos)}")
    cap = cv2.VideoCapture(str(raw_videos[0]))
    output_fps = float(cap.get(cv2.CAP_PROP_FPS) or source_fps)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()
    future = frames[history_count - 1 : history_count - 1 + future_count]
    if len(future) < future_count:
        raise RuntimeError(f"{sample_id} seed={seed}: only {len(future)} future frames")
    write_video(cv2, future, video_path, output_fps)
    for index, frame in enumerate(future):
        if not cv2.imwrite(str(frame_root / numeric_png_name(index)), frame):
            raise RuntimeError(f"{sample_id} seed={seed}: failed writing frame {index}")
    return {
        "sample_id": sample_id,
        "dataset_part": row.get("dataset_part", ""),
        "view_type": row.get("view_type", ""),
        "phase": row.get("phase", ""),
        "video": str(video_path),
        "status": "ok",
        "notes": f"seed={seed}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--sample_ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ltx_repo", type=Path, default=Path("/workspace/third_party/LTX-Video"))
    parser.add_argument("--ltx_python", type=Path, default=Path("/workspace/experiments/wan_ticft/ltxv_env/bin/python"))
    parser.add_argument("--pipeline_config", type=Path, default=Path("/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--prompt_mode", default="traffic_detail")
    parser.add_argument("--max_history_frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--future_frames", type=int, default=59)
    parser.add_argument("--image_cond_noise_scale", type=float, default=0.025)
    parser.add_argument("--vae_tiling", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    rows = read_rows(args.manifest, args.limit, None, "bdd", args.sample_ids)
    if not rows:
        raise RuntimeError("no rows selected")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = {seed: [] for seed in args.seeds}
    for row in rows:
        for seed in args.seeds:
            reports[seed].append(run_one(cv2, args, row, seed))
            write_report(args.out_dir / f"seed{seed}" / "report.csv", reports[seed])


if __name__ == "__main__":
    main()
