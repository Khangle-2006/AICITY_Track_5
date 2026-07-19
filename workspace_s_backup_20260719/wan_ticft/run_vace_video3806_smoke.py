"""Run a tiny Wan/VACE video-to-video smoke on video3806.

This is a deliberately narrow test: use the current best generated frame
sequence as VACE control, with low control scale, to see whether Wan can improve
realism without replacing the scene. It is not a submission pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


NEGATIVE_PROMPT = (
    "scene change, different road, different camera viewpoint, dashboard interior, "
    "duplicated cars, ghost cars, double exposure, afterimage, dark trails, warped lanes, "
    "cartoon, synthetic, blur, low quality"
)


def sorted_pngs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)


def read_frames(folder: Path, count: int, width: int, height: int) -> list[Image.Image]:
    paths = sorted_pngs(folder)
    if not paths:
        raise RuntimeError(f"no png frames in {folder}")
    selected = paths[:count]
    frames = []
    for path in selected:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot read {path}")
        rgb = cv2.cvtColor(cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb).convert("RGB"))
    while len(frames) < count:
        frames.append(frames[-1])
    return frames


def load_pipe() -> WanVideoPipeline:
    return WanVideoPipeline.from_pretrained(
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
                path="/workspace/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                skip_download=True,
            ),
        ],
        tokenizer_config=ModelConfig(
            path="/workspace/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
            skip_download=True,
        ),
        redirect_common_files=False,
    )


def make_compare(input_dir: Path, out_dir: Path, width: int, height: int, count: int) -> None:
    import imageio.v2 as imageio

    input_paths = sorted_pngs(input_dir)[:count]
    for mp4 in sorted((out_dir / "videos").glob("*.mp4")):
        cap = cv2.VideoCapture(str(mp4))
        frames = []
        idx = 0
        while idx < len(input_paths):
            ok, bgr = cap.read()
            if not ok:
                break
            inp = cv2.imread(str(input_paths[idx]), cv2.IMREAD_COLOR)
            inp = cv2.resize(inp, (width, height), interpolation=cv2.INTER_AREA)
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
            combo = np.concatenate([inp, bgr], axis=1)
            cv2.putText(combo, f"input f{idx}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(combo, mp4.stem, (width + 8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
            frames.append(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        if frames:
            imageio.mimsave(out_dir / f"compare_{mp4.stem}.mp4", frames, fps=8, quality=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--scales", default="0.15,0.30")
    parser.add_argument(
        "--prompt",
        default=(
            "Realistic dashcam traffic video restoration. Use the provided control video as the exact "
            "layout and motion. Preserve road geometry, vehicles, lanes, buildings, windshield, lighting, "
            "and camera viewpoint. Remove temporal ghosting, afterimages, duplicated cars, and dark road trails."
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.out_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    control = read_frames(args.input_dir, args.num_frames, args.width, args.height)
    reference = control[0]
    pipe = load_pipe()
    scales = [float(raw) for raw in args.scales.split(",") if raw.strip()]
    for offset, scale in enumerate(scales):
        label = f"vace_s{int(round(scale * 100)):02d}"
        out_path = video_dir / f"{label}.mp4"
        print(f"running {label}", flush=True)
        output = pipe(
            prompt=args.prompt,
            negative_prompt=NEGATIVE_PROMPT,
            vace_video=control,
            vace_reference_image=reference,
            vace_scale=scale,
            seed=args.seed + offset,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            tiled=True,
        )
        save_video(output, str(out_path), fps=8, quality=5)
        print(f"wrote {out_path}", flush=True)
    make_compare(args.input_dir, args.out_dir, args.width, args.height, args.num_frames)


if __name__ == "__main__":
    main()
