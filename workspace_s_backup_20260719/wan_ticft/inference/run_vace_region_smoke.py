from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image

from baseline.frame_io import numeric_pngs
from baseline.index_io import load_test_rows
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


NEGATIVE_PROMPT = (
    "camera change, scene change, altered road layout, distorted vehicles, "
    "duplicated objects, ghosting, blur, low quality, unrealistic motion"
)


def short_prompt(row: dict[str, str], max_words: int = 120) -> str:
    prefix = (
        "Realistic traffic video continuation. Preserve the same camera, road, "
        "background, lighting, vehicles, and pedestrians. "
    )
    text = " ".join(
        part.strip()
        for part in (row.get("caption_pedestrian", ""), row.get("caption_vehicle", ""))
        if part.strip()
    )
    return " ".join((prefix + text).split()[:max_words])


def load_frames(folder: Path, count: int, size: tuple[int, int]) -> list[Image.Image]:
    paths = numeric_pngs(folder)
    if not paths:
        raise RuntimeError(f"no numeric PNG frames found in {folder}")
    frames = [Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS) for path in paths[:count]]
    while len(frames) < count:
        frames.append(frames[-1].copy())
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=Path, default=Path("/workspace/experiments/index"))
    parser.add_argument("--region_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--sample", default="")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows = [row for row in load_test_rows(args.index_dir) if row.get("view_type") == "bdd"]
    if args.sample:
        rows = [row for row in rows if row.get("source_folder") == args.sample]
    if not rows:
        raise RuntimeError("no matching BDD row")
    row = rows[0]
    sample = row["source_folder"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.out_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    control = load_frames(args.region_dir / sample, args.num_frames, (args.width, args.height))
    history_paths = numeric_pngs(Path(row["input_dir"]))
    if not history_paths:
        raise RuntimeError(f"no history frames for {sample}")
    reference = Image.open(history_paths[-1]).convert("RGB").resize(
        (args.width, args.height), Image.Resampling.LANCZOS
    )

    print("Loading Wan2.1 VACE 1.3B...", flush=True)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                path="/workspace/models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
                skip_download=True,
            ),
            ModelConfig(
                path="/workspace/third_party/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors",
                skip_download=True,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="Wan2.1_VAE.pth",
                download_source="huggingface",
            ),
        ],
        tokenizer_config=ModelConfig(
            path="/workspace/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
            skip_download=True,
        ),
        redirect_common_files=False,
    )

    started = time.time()
    output = pipe(
        prompt=short_prompt(row),
        negative_prompt=NEGATIVE_PROMPT,
        vace_video=control,
        vace_reference_image=reference,
        vace_scale=1.0,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        tiled=True,
    )
    elapsed = time.time() - started
    output_path = video_dir / f"{sample}.mp4"
    save_video(output, str(output_path), fps=15, quality=5)

    report = {
        "sample": sample,
        "video": str(output_path),
        "status": "ok",
        "num_frames": args.num_frames,
        "steps": args.steps,
        "seed": args.seed,
        "elapsed_sec": elapsed,
    }
    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report.keys())
        writer.writeheader()
        writer.writerow(report)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
