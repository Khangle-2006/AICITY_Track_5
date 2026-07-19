"""Validate a DiffSynth Wan2.2-TI2V LoRA on a Track 5 manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_kv(key: str, value: object, width: int = 42) -> None:
    print(f"{key:<{width}} {value}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_condition_image(row: dict[str, Any], policy: str) -> str:
    frames = row.get("frames") or {}
    condition = frames.get("condition") or []
    if not condition:
        raise ValueError("row has no condition frames")
    if policy == "first":
        return str(condition[0])
    if policy == "middle":
        return str(condition[len(condition) // 2])
    return str(condition[-1])


def prompt_for(row: dict[str, Any], max_words: int) -> str:
    prompt = str(row.get("prompt") or "").strip()
    if not prompt:
        prompt = f"{row.get('view_type', 'traffic camera')} traffic scene, realistic near future video"
    words = prompt.split()
    if max_words > 0 and len(words) > max_words:
        prompt = " ".join(words[:max_words])
    return prompt


def safe_name(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "sample"


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = [
        "idx",
        "sample_id",
        "dataset_part",
        "view_type",
        "phase",
        "status",
        "seed",
        "prompt_words",
        "elapsed_sec",
        "image",
        "video",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    by_status: dict[str, int] = {}
    by_view: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        view = row.get("view_type") or "(unknown)"
        by_view[view] = by_view.get(view, 0) + 1
    summary = {
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "lora": str(args.lora),
        "rows": len(rows),
        "by_status": by_status,
        "by_view": by_view,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "lora_alpha": args.lora_alpha,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_pipe(args: argparse.Namespace):
    import torch
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.2_VAE.pth"),
        ],
    )
    pipe.load_lora(pipe.dit, str(args.lora), alpha=args.lora_alpha)
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--model_id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=13)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--max_prompt_words", type=int, default=120)
    parser.add_argument("--image_policy", choices=["first", "middle", "last"], default="last")
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    args.lora = args.lora.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.manifest)
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]

    print_header("DIFFSYNTH WAN2.2 TI2V LORA VALIDATION")
    print_kv("manifest", args.manifest)
    print_kv("out_dir", args.out_dir)
    print_kv("lora", args.lora)
    print_kv("rows", len(rows))
    print_kv("size", f"{args.width}x{args.height}")
    print_kv("num_frames", args.num_frames)
    print_kv("steps", args.steps)
    print_kv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))

    # Import after argument parsing so this script remains importable without DiffSynth installed.
    from PIL import Image
    from diffsynth.utils.data import save_video

    pipe = load_pipe(args)
    report: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample_id = safe_name(str(row.get("sample_id") or f"sample_{idx:06d}"))
        image_path = select_condition_image(row, args.image_policy)
        prompt = prompt_for(row, args.max_prompt_words)
        seed = args.seed + idx
        out_video = video_dir / f"{idx:06d}_{sample_id}.mp4"
        result = {
            "idx": idx,
            "sample_id": row.get("sample_id", ""),
            "dataset_part": row.get("dataset_part", ""),
            "view_type": row.get("view_type", ""),
            "phase": row.get("phase", ""),
            "status": "failed",
            "seed": seed,
            "prompt_words": len(prompt.split()),
            "elapsed_sec": "0.00",
            "image": image_path,
            "video": str(out_video),
            "error": "",
        }
        print_kv(f"[{idx + 1}/{len(rows)}] sample", sample_id)
        start = time.time()
        try:
            input_image = Image.open(image_path).convert("RGB")
            video = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                input_image=input_image,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                seed=seed,
                tiled=True,
            )
            save_video(video, str(out_video), fps=args.fps, quality=5)
            result["status"] = "ok" if out_video.exists() and out_video.stat().st_size > 0 else "failed"
        except Exception as exc:
            result["error"] = repr(exc)
        result["elapsed_sec"] = f"{time.time() - start:.2f}"
        print_kv("status", f"{result['status']} | {result['elapsed_sec']} sec")
        report.append(result)
        write_report(args.out_dir / "report.csv", report)
        write_summary(args.out_dir / "summary.json", report, args)

    print_header("RESULT")
    print_kv("report", args.out_dir / "report.csv")
    print_kv("summary", args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
