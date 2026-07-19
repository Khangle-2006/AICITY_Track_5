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


def numeric_png_name(index: int) -> str:
    return f"{index}.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--sample_id", default="test_bdd_pc_5k_bdd_video3806_phase2")
    parser.add_argument("--ltx_repo", type=Path, default=Path("/workspace/third_party/LTX-Video"))
    parser.add_argument("--ltx_python", type=Path, default=Path("/workspace/experiments/wan_ticft/ltxv_env/bin/python"))
    parser.add_argument("--pipeline_config", type=Path, default=Path("/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--prompt_mode", default="traffic_detail")
    parser.add_argument("--prompt_override", default="")
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
    rows = read_rows(args.manifest, 0, None, "bdd", [args.sample_id])
    if not rows:
        raise RuntimeError(f"missing sample {args.sample_id}")
    row = rows[0]
    sample_dir_name = "video3806"
    condition = (row.get("frame_indices") or {}).get("condition") or []
    last_index = int(condition[-1])
    history_count = compatible_history_length(last_index + 1, args.max_history_frames)
    if not history_count:
        raise RuntimeError("not enough history")

    try:
        history, source_fps = read_history(cv2, Path(row["video_path"]), last_index, history_count)
        decode_mode = "random_seek"
    except RuntimeError:
        history, source_fps = read_history_sequential(cv2, Path(row["video_path"]), last_index, history_count)
        decode_mode = "sequential"
    condition_paths = (row.get("frames") or {}).get("condition") or []
    reference = read_rgb(cv2, Path(condition_paths[-1])) if condition_paths else None
    history, orientation, mean_error = align_history_to_reference(cv2, history, reference)
    print(
        f"history={history_count} decode={decode_mode}/{orientation} diff={mean_error:.3f} fps={source_fps:.3f}",
        flush=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.out_dir / "_inputs" / "history.mp4"
    if not (args.resume and input_path.exists()):
        write_video(cv2, history, input_path, source_fps)

    rows_report = []
    for seed in args.seeds:
        label = f"seed{seed}"
        frame_root = args.out_dir / label / sample_dir_name
        raw_dir = args.out_dir / "_raw" / label
        video_path = args.out_dir / label / "future.mp4"
        if args.resume and frame_root.exists() and len(list(frame_root.glob("*.png"))) >= args.future_frames:
            print(f"{label}: resume", flush=True)
            rows_report.append({"seed": seed, "frames": str(frame_root), "video": str(video_path), "status": "resume"})
            continue
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        if frame_root.exists():
            shutil.rmtree(frame_root)
        raw_dir.mkdir(parents=True, exist_ok=True)
        frame_root.mkdir(parents=True, exist_ok=True)
        prompt = args.prompt_override.strip() or short_prompt(row, args.prompt_mode)
        command = [
            str(args.ltx_python),
            str(args.ltx_repo / "inference.py"),
            "--prompt",
            prompt,
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
        print(f"{label}: running", flush=True)
        subprocess.run(command, check=True, env=env)
        raw_videos = sorted(raw_dir.glob("*.mp4"))
        if len(raw_videos) != 1:
            raise RuntimeError(f"{label}: expected one raw mp4, found {len(raw_videos)}")
        cap = cv2.VideoCapture(str(raw_videos[0]))
        output_fps = float(cap.get(cv2.CAP_PROP_FPS) or source_fps)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
        cap.release()
        future = frames[history_count - 1 : history_count - 1 + args.future_frames]
        if len(future) < args.future_frames:
            raise RuntimeError(f"{label}: only {len(future)} future frames")
        write_video(cv2, future, video_path, output_fps)
        for index, frame in enumerate(future):
            if not cv2.imwrite(str(frame_root / numeric_png_name(index)), frame):
                raise RuntimeError(f"{label}: failed writing frame {index}")
        print(f"{label}: wrote {frame_root}", flush=True)
        rows_report.append({"seed": seed, "frames": str(frame_root), "video": str(video_path), "status": "ok"})

    with (args.out_dir / "seed_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "frames", "video", "status"])
        writer.writeheader()
        writer.writerows(rows_report)


if __name__ == "__main__":
    main()
