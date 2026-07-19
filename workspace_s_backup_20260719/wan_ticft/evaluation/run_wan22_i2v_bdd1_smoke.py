from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image


def read_rows(path: Path, sample_id: str | None):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if sample_id is None or row.get("sample_id") == sample_id:
                yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()

    repo = Path("/workspace/third_party/DiffSynth-Studio")
    sys.path.insert(0, str(repo))
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import save_video

    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.out_dir / "videos"
    anchors_dir = args.out_dir / "anchors"
    videos_dir.mkdir(parents=True, exist_ok=True)
    anchors_dir.mkdir(parents=True, exist_ok=True)

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
    pipe = WanVideoPipeline.from_pretrained(
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

    rows = list(read_rows(args.manifest, args.sample_id))
    if not rows:
        raise RuntimeError("no rows selected")

    report_rows = []
    prompt = (
        "Dashcam driving video continuation from the exact input frame. Preserve the same camera, road geometry, "
        "lane markings, parked cars, buildings, sidewalks, lighting, object identities, scale, and positions. "
        "Only natural forward time motion, no new objects, no scene redesign."
    )
    negative = (
        "new scene, camera angle change, hallucinated vehicle, duplicated pedestrian, object replacement, "
        "warped road, lane drift, double exposure, ghosting, smear, synthetic texture, dreamy blur, flicker, "
        "low quality, text, watermark"
    )
    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        cond_paths = (row.get("frames") or {}).get("condition") or []
        if not cond_paths:
            report_rows.append({"sample_id": sample_id, "video": "", "status": "bad", "notes": "missing_condition"})
            continue
        anchor = Image.open(cond_paths[-1]).convert("RGB").resize((args.width, args.height), Image.BICUBIC)
        anchor_path = anchors_dir / f"{sample_id}.jpg"
        anchor.save(anchor_path, quality=95)
        out_path = videos_dir / f"{sample_id}.mp4"
        video = pipe(
            prompt=prompt,
            negative_prompt=negative,
            seed=args.seed + index,
            tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            input_image=anchor,
            switch_DiT_boundary=0.9,
        )
        save_video(video, str(out_path), fps=15, quality=5)
        report_rows.append({"sample_id": sample_id, "video": str(out_path), "status": "ok", "notes": str(anchor_path)})

    with (args.out_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "notes"])
        writer.writeheader()
        writer.writerows(report_rows)
    print(json.dumps({"status": "ok", "out_dir": str(args.out_dir), "rows": len(report_rows)}, indent=2))


if __name__ == "__main__":
    main()
