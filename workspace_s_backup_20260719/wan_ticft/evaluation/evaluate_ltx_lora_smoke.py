from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from PIL import Image

from baseline.frame_io import import_cv2_np
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
from ltxv_trainer.model_loader import load_ltxv_components


def select_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("view_type") == "bdd":
                rows.append(row)
                if len(rows) == limit:
                    break
    return rows


def read_condition_frame(row: dict) -> Image.Image:
    condition_paths = (row.get("frames") or {}).get("condition") or []
    if not condition_paths:
        raise RuntimeError(f"{row.get('sample_id')}: missing materialized condition frames")
    # Use the already-materialized condition frame. The source BDD videos can
    # require rotation/alignment; the manifest frame path is the canonical
    # orientation used by the rest of the evaluation pipeline.
    return Image.open(condition_paths[-1]).convert("RGB")


def generate(pipeline, image, prompt: str, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = pipeline(
            image=image,
            prompt=prompt,
            negative_prompt="blurry, jittery, distorted, inconsistent geometry",
            width=512,
            height=288,
            num_frames=121,
            frame_rate=30,
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
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    rows = select_rows(args.manifest, args.limit)
    if len(rows) != args.limit:
        raise RuntimeError(f"found only {len(rows)} BDD rows")
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

    base_report = []
    lora_report = []
    selected = []
    prompt = (
        "Continue the observed forward-facing driving camera video. Preserve the road layout, lighting, camera "
        "viewpoint, vehicle identities, and pedestrian identities. Generate realistic ego motion and temporally "
        "consistent near-future traffic motion."
    )

    # Generate base outputs first so both candidates use identical model state,
    # samples, prompts, and seeds.
    inputs = []
    for position, row in enumerate(rows, start=1):
        image = read_condition_frame(row)
        inputs.append((row, image))
        output = args.out_dir / "base" / f"{row['sample_id']}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"base [{position}/{len(rows)}] {row['sample_id']}", flush=True)
        fps = float((row.get("frame_indices") or {}).get("fps") or 30.0)
        export_to_video(generate(pipeline, image, prompt, 100 + position), str(output), fps=round(fps))
        base_report.append({"sample_id": row["sample_id"], "video": str(output), "status": "ok"})

    pipeline.load_lora_weights(str(args.checkpoint.parent), weight_name=args.checkpoint.name)
    for position, (row, image) in enumerate(inputs, start=1):
        output = args.out_dir / "lora" / f"{row['sample_id']}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"lora [{position}/{len(rows)}] {row['sample_id']}", flush=True)
        fps = float((row.get("frame_indices") or {}).get("fps") or 30.0)
        export_to_video(generate(pipeline, image, prompt, 100 + position), str(output), fps=round(fps))
        lora_report.append({"sample_id": row["sample_id"], "video": str(output), "status": "ok"})
        selected.append(row)

    write_report(args.out_dir / "base_report.csv", base_report)
    write_report(args.out_dir / "lora_report.csv", lora_report)
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print({"rows": len(selected), "status": "ok"})


if __name__ == "__main__":
    main()
