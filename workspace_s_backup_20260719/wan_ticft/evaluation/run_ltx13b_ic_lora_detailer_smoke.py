"""Run official LTX 13B 0.9.8 IC-LoRA Detailer over frozen BDD reference predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video

from wan_ticft.evaluation.run_sa2va_actor_aware_depth_gate import read_video


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def report_video(path: Path, sample_id: str) -> Path:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("sample_id") == sample_id and row.get("status") == "ok":
                return Path(row["video"])
    raise RuntimeError(f"missing {sample_id} in {path}")


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source_report", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--lora_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--sample_id", action="append", required=True)
    parser.add_argument("--lora_scale", type=float, action="append", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--frames", type=int, default=41)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
    from ltxv_trainer.model_loader import load_ltxv_components

    order = {sample_id: index for index, sample_id in enumerate(args.sample_id)}
    rows = [row for row in read_jsonl(args.manifest) if row.get("sample_id") in order]
    rows.sort(key=lambda row: order[row["sample_id"]])
    if len(rows) != len(args.sample_id):
        raise RuntimeError(f"expected {len(args.sample_id)} manifest rows, found {len(rows)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "selected_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print("Loading LTX 13B components on CPU...", flush=True)
    components = load_ltxv_components(
        model_source=str(args.model_path),
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
    )
    pipeline.load_lora_weights(
        str(args.lora_path.parent),
        weight_name=args.lora_path.name,
        adapter_name="detailer",
    )
    if hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    pipeline.enable_model_cpu_offload(gpu_id=0)
    pipeline.set_progress_bar_config(disable=False)

    prompt = (
        "Photorealistic forward-facing driving camera video. Preserve the reference video motion, exact road and "
        "building geometry, pedestrian identity, vehicle trajectories, camera motion, and timing. Restore only "
        "natural fine texture and sharp stable background detail without changing composition or adding objects."
    )
    negative = (
        "changed geometry, changed motion, extra people, extra vehicles, duplicated objects, hallucination, "
        "jitter, flicker, warped edges, blur, oversharpening"
    )
    reports = {f"detailer_s{int(round(scale * 100)):03d}": [] for scale in args.lora_scale}
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        source_path = report_video(args.source_report, sample_id)
        source_frames, fps = read_video(source_path)
        if len(source_frames) < args.frames:
            raise RuntimeError(f"{sample_id}: reference has {len(source_frames)} frames, require {args.frames}")
        reference = torch.from_numpy(np.stack(source_frames[: args.frames]).astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        for scale in args.lora_scale:
            tag = f"detailer_s{int(round(scale * 100)):03d}"
            output = args.out_dir / tag / f"{sample_id}.mp4"
            if output.is_file():
                print(f"[{position}/{len(rows)}] reuse {tag} {sample_id}", flush=True)
            else:
                generator = torch.Generator(device="cuda").manual_seed(args.seed)
                print(f"[{position}/{len(rows)}] generating {tag} {sample_id}...", flush=True)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = pipeline(
                        reference_video=reference,
                        output_reference_comparison=False,
                        prompt=prompt,
                        negative_prompt=negative,
                        width=args.width,
                        height=args.height,
                        num_frames=args.frames,
                        frame_rate=float(fps or 30.0),
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=generator,
                        attention_kwargs={"scale": scale},
                        output_type="np",
                    )
                frames = result.frames[0]
                output.parent.mkdir(parents=True, exist_ok=True)
                export_to_video(frames, str(output), fps=float(fps or 30.0))
                del result, frames
                torch.cuda.empty_cache()
            reports[tag].append({"sample_id": sample_id, "video": str(output), "status": "ok"})
            write_report(args.out_dir / tag / "report.csv", reports[tag])
        del reference, source_frames


if __name__ == "__main__":
    main()
