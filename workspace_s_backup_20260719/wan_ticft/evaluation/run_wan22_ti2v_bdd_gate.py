from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from baseline.frame_io import import_cv2_np, read_rgb
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


NEGATIVE_PROMPT = (
    "scene change, camera angle change, dashboard interior, different road, new city, "
    "hallucinated objects, duplicated vehicles, duplicated pedestrians, warped geometry, "
    "ghosting, blur, double exposure, low quality, text, watermark"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def resize_rgb(cv2, img, width: int, height: int):
    if img.shape[1] == width and img.shape[0] == height:
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def load_anchor_image(cv2, row: dict[str, Any], width: int, height: int) -> Image.Image:
    paths = (row.get("frames") or {}).get("condition") or []
    if not paths:
        raise RuntimeError(f"{row.get('sample_id')}: missing condition frames")
    img = read_rgb(cv2, Path(paths[-1]))
    if img is None:
        raise RuntimeError(f"{row.get('sample_id')}: cannot read condition frame")
    return Image.fromarray(resize_rgb(cv2, img, width, height)).convert("RGB")


def prompt_for(row: dict[str, Any], max_words: int = 110) -> str:
    base = (
        "Forward-facing driving camera video. Continue the exact same road scene from "
        "the reference image. Preserve camera viewpoint, horizon, road geometry, "
        "buildings, vehicles, pedestrians, windshield, dashboard, lighting, and weather. "
        "Only advance existing motion naturally. Keep sharp realistic video, no blur, "
        "no ghosting, no scene replacement. "
    )
    text = str(row.get("prompt") or "")
    if not text:
        text = " ".join(
            str(row.get(key, "")).strip()
            for key in ("caption_vehicle", "caption_pedestrian")
            if str(row.get(key, "")).strip()
        )
    return " ".join((base + text).split()[:max_words])


def load_pipe(args) -> WanVideoPipeline:
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
    converted = Path("/workspace/third_party/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors")
    diffusion = Path("/workspace/third_party/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B")
    diffusion_shards = sorted(diffusion.glob("diffusion_pytorch_model-*.safetensors"))
    if not diffusion_shards:
        raise FileNotFoundError(f"missing Wan2.2 diffusion shards in {diffusion}")
    print("Loading Wan2.2-TI2V-5B...", flush=True)
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(converted / "models_t5_umt5-xxl-enc-bf16.safetensors"), skip_download=True, **vram_config),
            ModelConfig(path=[str(path) for path in diffusion_shards], skip_download=True, **vram_config),
            ModelConfig(path=str(converted / "Wan2.2_VAE.safetensors"), skip_download=True, **vram_config),
        ],
        tokenizer_config=ModelConfig(
            path="/workspace/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
            skip_download=True,
        ),
        vram_limit=args.vram_limit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B BDD image-anchor smoke/gate.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=41)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--vram_limit", type=float, default=46.0)
    args = parser.parse_args()

    cv2, _np = import_cv2_np()
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"][: args.limit]
    if not rows:
        raise RuntimeError("no BDD rows")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "selected_manifest.jsonl", rows)

    pipe = load_pipe(args)
    report_rows = []
    for pos, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        out_path = video_dir / f"{sample_id}.mp4"
        started = time.time()
        try:
            video = pipe(
                prompt=prompt_for(row),
                negative_prompt=NEGATIVE_PROMPT,
                input_image=load_anchor_image(cv2, row, args.width, args.height),
                seed=args.seed + pos - 1,
                tiled=True,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
            )
            save_video(video, str(out_path), fps=30, quality=5)
            status = "ok"
            notes = ""
        except Exception as exc:  # noqa: BLE001
            status = "bad"
            notes = repr(exc)
        elapsed = time.time() - started
        report_rows.append(
            {
                "sample_id": sample_id,
                "video": str(out_path),
                "status": status,
                "source_fps": "30.000000",
                "future_frames": args.num_frames,
                "elapsed_sec": f"{elapsed:.3f}",
                "notes": notes,
            }
        )
        print(f"[{pos}/{len(rows)}] {sample_id} | {status} elapsed={elapsed:.1f}s", flush=True)

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "video", "status", "source_fps", "future_frames", "elapsed_sec", "notes"],
        )
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"output={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
