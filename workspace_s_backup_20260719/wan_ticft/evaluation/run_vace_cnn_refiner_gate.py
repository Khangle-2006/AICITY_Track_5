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
from wan_ticft.evaluation.evaluate_video_reports import read_report, read_video_rgb_frames, video_fps


NEGATIVE_PROMPT = (
    "scene change, camera angle change, dashboard interior view, new road, new city, "
    "object replacement, hallucinated pedestrians, duplicated vehicles, ghosting, blur, "
    "double exposure, warped geometry, low quality"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def prompt_for(row: dict[str, Any], max_words: int = 90) -> str:
    base = (
        "Realistic forward-facing driving video restoration. Use the provided control "
        "video as the exact motion/layout guide. Preserve road geometry, buildings, "
        "vehicles, pedestrians, windshield, lighting, and camera viewpoint. Improve "
        "natural video texture and temporal consistency without changing the scene. "
    )
    text = str(row.get("prompt") or "")
    if not text:
        text = " ".join(
            str(row.get(key, "")).strip()
            for key in ("caption_vehicle", "caption_pedestrian")
            if str(row.get(key, "")).strip()
        )
    return " ".join((base + text).split()[:max_words])


def resize_rgb(cv2, img, width: int, height: int):
    if img.shape[1] == width and img.shape[0] == height:
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def rgb_to_pil(img) -> Image.Image:
    return Image.fromarray(img).convert("RGB")


def load_control_frames(cv2, video: Path, count: int, width: int, height: int) -> list[Image.Image]:
    fps = video_fps(cv2, video) or 30.0
    frames = read_video_rgb_frames(cv2, video, list(range(count)))
    if not frames:
        raise RuntimeError(f"cannot read control video: {video}")
    ordered = []
    for index in range(count):
        frame = frames.get(index)
        if frame is None:
            frame = ordered[-1] if ordered else next(iter(frames.values()))
        ordered.append(rgb_to_pil(resize_rgb(cv2, frame, width, height)))
    return ordered


def load_reference(cv2, row: dict[str, Any], width: int, height: int) -> Image.Image:
    paths = (row.get("frames") or {}).get("condition") or []
    if not paths:
        raise RuntimeError(f"{row.get('sample_id')}: missing condition frames")
    img = read_rgb(cv2, Path(paths[-1]))
    if img is None:
        raise RuntimeError(f"{row.get('sample_id')}: cannot read {paths[-1]}")
    return rgb_to_pil(resize_rgb(cv2, img, width, height))


def load_pipe() -> WanVideoPipeline:
    print("Loading Wan2.1 VACE 1.3B...", flush=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Use Wan/VACE as a generative refiner over CNN/LTX videos.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=41)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vace_scales", default="0.5,0.8")
    args = parser.parse_args()

    cv2, _np = import_cv2_np()
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"][: args.limit]
    if not rows:
        raise RuntimeError("no BDD rows")
    base_report = read_report(args.base_report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "selected_manifest.jsonl", rows)

    scales = [float(raw) for raw in args.vace_scales.split(",") if raw.strip()]
    reports: dict[str, list[dict[str, Any]]] = {f"vace_s{int(scale * 100):02d}": [] for scale in scales}
    pipe = load_pipe()

    for pos, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        base_video = base_report.get(sample_id)
        if base_video is None:
            print(f"[{pos}/{len(rows)}] {sample_id} | skipped: missing base", flush=True)
            continue
        control = load_control_frames(cv2, base_video, args.num_frames, args.width, args.height)
        reference = load_reference(cv2, row, args.width, args.height)
        prompt = prompt_for(row)
        for scale in scales:
            label = f"vace_s{int(scale * 100):02d}"
            video_dir = args.out_dir / label / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            out_path = video_dir / f"{sample_id}.mp4"
            started = time.time()
            try:
                output = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    vace_video=control,
                    vace_reference_image=reference,
                    vace_scale=scale,
                    seed=args.seed + pos,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.steps,
                    tiled=True,
                )
                save_video(output, str(out_path), fps=30, quality=5)
                status = "ok"
                notes = ""
            except Exception as exc:  # noqa: BLE001
                status = "bad"
                notes = repr(exc)
            elapsed = time.time() - started
            reports[label].append(
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
            print(
                f"[{pos}/{len(rows)}] {sample_id} | {label} {status} elapsed={elapsed:.1f}s",
                flush=True,
            )

    for label, report_rows in reports.items():
        report_path = args.out_dir / label / "report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "video", "status", "source_fps", "future_frames", "elapsed_sec", "notes"],
            )
            writer.writeheader()
            writer.writerows(report_rows)
    print(f"output={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
