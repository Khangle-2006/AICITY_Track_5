"""Refine videos with LTX hard video-conditioning instead of reference guidance.

The LTXConditionPipeline treats ``video=...`` at frame_index=0 as actual latent
initialization/conditioning. With strength close to 1.0, the scaffold geometry
stays locked and only the final denoising steps can alter texture. This is a
small gate for denoise-around-scaffold, not free video regeneration.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from PIL import Image

from baseline.frame_io import import_cv2_np
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline, LTXVideoCondition
from ltxv_trainer.model_loader import load_ltxv_components


def read_video(cv2, path: Path, width: int, height: int):
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def to_pil(frames):
    return [Image.fromarray(frame) for frame in frames]


def compatible_segment(frames, chunk: int):
    seg = list(frames[:chunk])
    original = len(seg)
    if original < 2:
        return seg, original
    while len(seg) < chunk:
        seg.append(seg[-1])
    return seg, original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--chunk", type=int, default=9)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--strength", type=float, default=0.92)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prompt", default="Realistic BDD dashcam video. Preserve exact road geometry, vehicles, pedestrians, lanes, camera motion and timing.")
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    components = load_ltxv_components(
        model_source="LTXV_2B_0.9.6_DEV",
        load_text_encoder_in_8bit=False,
        transformer_dtype=torch.bfloat16,
        vae_dtype=torch.bfloat16,
    )
    pipe = LTXConditionPipeline(
        scheduler=components.scheduler,
        vae=components.vae,
        text_encoder=components.text_encoder,
        tokenizer=components.tokenizer,
        transformer=components.transformer,
    ).to("cuda")
    if args.checkpoint:
        pipe.load_lora_weights(str(args.checkpoint.parent), weight_name=args.checkpoint.name)
    pipe.set_progress_bar_config(disable=True)

    videos = [path for path in sorted(args.in_dir.glob("*")) if path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
    if args.limit:
        videos = videos[: args.limit]
    rows = []
    for index, video in enumerate(videos, 1):
        frames, fps = read_video(cv2, video, args.width, args.height)
        refined = []
        for start in range(0, len(frames), args.chunk):
            segment, keep = compatible_segment(frames[start : start + args.chunk], args.chunk)
            if keep < 2:
                refined.extend(segment[:keep])
                continue
            condition = LTXVideoCondition(video=to_pil(segment), frame_index=0, strength=args.strength)
            generator = torch.Generator(device="cuda").manual_seed(args.seed + start)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = pipe(
                    conditions=[condition],
                    prompt=args.prompt,
                    negative_prompt="cartoon, synthetic, distorted geometry, duplicated cars, duplicated pedestrians, smeared lanes, light trails",
                    width=args.width,
                    height=args.height,
                    num_frames=len(segment),
                    frame_rate=int(round(fps)),
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    image_cond_noise_scale=0.0,
                    generator=generator,
                )
            refined.extend(list(output.frames[0])[:keep])
        out_path = args.out_dir / f"{video.stem}.mp4"
        export_to_video(refined[: len(frames)], str(out_path), fps=int(round(fps)))
        rows.append({"sample_id": video.stem, "video": str(out_path), "status": "ok", "source_fps": f"{fps:.6f}", "future_frames": len(refined[: len(frames)])})
        print(f"[{index}/{len(videos)}] {video.stem} | frames={len(frames)} out={len(refined[:len(frames)])}", flush=True)

    with (args.out_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "source_fps", "future_frames"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
