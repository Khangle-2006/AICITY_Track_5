"""Compare base LTX and IC-LoRA using the same observed-history reference video."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from diffusers.utils import export_to_video

from baseline.frame_io import import_cv2_np, read_rgb
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
from ltxv_trainer.model_loader import load_ltxv_components


def select_rows(path: Path, per_phase: int) -> list[dict]:
    selected = []
    counts = {str(phase): 0 for phase in range(5)}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            phase = str(row.get("phase", ""))
            if row.get("view_type") == "bdd" and phase in counts and counts[phase] < per_phase:
                selected.append(row)
                counts[phase] += 1
    if any(count != per_phase for count in counts.values()):
        raise RuntimeError(f"insufficient balanced BDD rows: {counts}")
    return selected


def read_rgb_frames(cv2, video_path: Path, indices: list[int]):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"cannot read {video_path} at frame {index}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def read_condition_frames(cv2, row: dict):
    paths = [Path(path) for path in (row.get("frames") or {}).get("condition", [])]
    frames = [read_rgb(cv2, path) for path in paths]
    if paths and all(frame is not None for frame in frames):
        return frames
    indices = [int(index) for index in row["frame_indices"]["condition"]]
    return read_rgb_frames(cv2, Path(row["video_path"]), indices)


def spread_reference(cv2, np, frames, count: int, width: int = 512, height: int = 288) -> torch.Tensor:
    resized = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]
    spread = [resized[min(len(resized) - 1, index * len(resized) // count)] for index in range(count)]
    array = np.stack(spread).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(0, 3, 1, 2)


def generate(pipeline, reference_video: torch.Tensor, prompt: str, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = pipeline(
            reference_video=reference_video,
            output_reference_comparison=False,
            prompt=prompt,
            negative_prompt="blurry, jittery, distorted, inconsistent geometry",
            width=512,
            height=288,
            num_frames=121,
            frame_rate=24,
            num_inference_steps=8,
            guidance_scale=1.0,
            generator=generator,
        )
    return result.frames[0]


def write_report(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--per_phase", type=int, default=1)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows = select_rows(args.manifest, args.per_phase)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    components = load_ltxv_components(
        model_source="LTXV_2B_0.9.6_DISTILLED",
        load_text_encoder_in_8bit=False,
        transformer_dtype=torch.bfloat16,
        vae_dtype=torch.bfloat16,
    )
    pipeline = LTXConditionPipeline(
        scheduler=components.scheduler,
        vae=components.vae,
        text_encoder=components.text_encoder,
        tokenizer=components.tokenizer,
        transformer=components.transformer,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=False)

    prompt = (
        "Continue the observed forward-facing driving camera video into the near future. "
        "Preserve camera motion, road geometry, object identity, and realistic traffic motion."
    )
    inputs = []
    base_report = []
    lora_report = []
    for position, row in enumerate(rows, start=1):
        frames = read_condition_frames(cv2, row)
        reference = spread_reference(cv2, np, frames, 121)
        inputs.append((row, reference))
        output = args.out_dir / "base" / f"{row['sample_id']}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"base [{position}/{len(rows)}] phase={row['phase']} {row['sample_id']}", flush=True)
        export_to_video(generate(pipeline, reference, prompt, 200 + position), str(output), fps=24)
        base_report.append({"sample_id": row["sample_id"], "video": str(output), "status": "ok"})

    pipeline.load_lora_weights(str(args.checkpoint.parent), weight_name=args.checkpoint.name)
    for position, (row, reference) in enumerate(inputs, start=1):
        output = args.out_dir / "ic_lora" / f"{row['sample_id']}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"ic_lora [{position}/{len(rows)}] phase={row['phase']} {row['sample_id']}", flush=True)
        export_to_video(generate(pipeline, reference, prompt, 200 + position), str(output), fps=24)
        lora_report.append({"sample_id": row["sample_id"], "video": str(output), "status": "ok"})

    write_report(args.out_dir / "base_report.csv", base_report)
    write_report(args.out_dir / "ic_lora_report.csv", lora_report)
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print({"rows": len(rows), "status": "ok"})


if __name__ == "__main__":
    main()
