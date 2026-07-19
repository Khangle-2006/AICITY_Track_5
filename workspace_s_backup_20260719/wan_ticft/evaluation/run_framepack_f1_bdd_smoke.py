from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
from diffusers.utils import export_to_video
from PIL import Image
from transformers import SiglipImageProcessor, SiglipVisionModel


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Small FramePack F1 BDD image-to-video smoke.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--guidance", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--row_prompt_chars", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.out_dir / "videos"
    input_dir = args.out_dir / "inputs"
    video_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.manifest, args.limit)
    if not rows:
        raise RuntimeError("no BDD rows found")

    prompt_suffix = (
        " Continue the exact same forward-facing dashcam driving scene. Preserve camera viewpoint, "
        "road geometry, dashboard, lighting, vehicles, pedestrians, signs, buildings, and object identities. "
        "Only realistic near-future traffic motion should change. Avoid ghosting, double exposure, smearing, "
        "melting, camera cuts, and new objects."
    )

    transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
        "lllyasviel/FramePack_F1_I2V_HY_20250503",
        torch_dtype=torch.bfloat16,
    )
    feature_extractor = SiglipImageProcessor.from_pretrained(
        "lllyasviel/flux_redux_bfl",
        subfolder="feature_extractor",
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        "lllyasviel/flux_redux_bfl",
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    pipe = HunyuanVideoFramepackPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        transformer=transformer,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    report = []
    for pos, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        cond_paths = (row.get("frames") or {}).get("condition") or []
        if not cond_paths:
            continue
        image = Image.open(cond_paths[-1]).convert("RGB")
        image.save(input_dir / f"{sample_id}.png")
        row_prompt = (row.get("prompt") or "")[: max(0, args.row_prompt_chars)]
        prompt = (row_prompt + " " + prompt_suffix).strip()
        generator = torch.Generator().manual_seed(args.seed + pos)
        frames = pipe(
            image=image,
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            sampling_type="vanilla",
        ).frames[0]
        out_video = video_dir / f"{sample_id}.mp4"
        export_to_video(frames, str(out_video), fps=16)
        report.append({"sample_id": sample_id, "video": str(out_video), "status": "ok", "future_frames": len(frames)})
        print(f"[{pos}/{len(rows)}] {sample_id} frames={len(frames)} out={out_video}", flush=True)

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "future_frames"])
        writer.writeheader()
        writer.writerows(report)
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"report={args.out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
