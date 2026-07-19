from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


NEGATIVE = (
    "overexposed, static, blurry, low quality, jpeg artifacts, distorted geometry, "
    "ghosting, double exposure, smear, melting, camera cut, new scene, wrong viewpoint, "
    "extra vehicles, deformed cars, deformed pedestrians, text, watermark"
)

PROMPT = (
    "Forward-facing dashcam video prediction. Continue the same road scene from the input image. "
    "Preserve camera viewpoint, road layout, dashboard, lighting, buildings, traffic lights, vehicles, "
    "pedestrians, signs, and object identities. Only realistic near-future traffic motion changes."
)


def read_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("view_type") != "bdd":
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def build_pipe() -> WanVideoPipeline:
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.2-I2V-A14B",
                origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-I2V-A14B",
                origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-I2V-A14B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                **vram_config,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-I2V-A14B",
                origin_file_pattern="Wan2.1_VAE.pth",
                **vram_config,
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024**3) - 2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan2.2 I2V-A14B BDD smoke.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.out_dir / "videos"
    input_dir = args.out_dir / "inputs"
    video_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.manifest, args.limit)
    if not rows:
        raise RuntimeError("no BDD rows")

    pipe = build_pipe()
    report_rows = []
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        condition = (row.get("frames") or {}).get("condition") or []
        if not condition:
            continue
        image = Image.open(condition[-1]).convert("RGB").resize((args.width, args.height))
        image.save(input_dir / f"{sample_id}.png")
        video = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            input_image=image,
            seed=args.seed + position,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            tiled=True,
            switch_DiT_boundary=0.9,
        )
        out_video = video_dir / f"{sample_id}.mp4"
        save_video(video, str(out_video), fps=15, quality=5)
        report_rows.append(
            {
                "sample_id": sample_id,
                "video": str(out_video),
                "status": "ok",
                "future_frames": len(video),
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "cfg_scale": args.cfg_scale,
            }
        )
        print(f"[{position}/{len(rows)}] {sample_id} frames={len(video)} out={out_video}", flush=True)

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "video", "status", "future_frames", "width", "height", "steps", "cfg_scale"],
        )
        writer.writeheader()
        writer.writerows(report_rows)
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"report={args.out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
