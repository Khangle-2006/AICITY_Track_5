"""Prepare paired WTS base/target metadata and a bounded LTX IC-LoRA smoke config."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import yaml


DEFAULT_PROMPT = "Restore this traffic prediction video while preserving exact camera geometry and actor motion."


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_report(path: Path) -> dict[str, Path]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["sample_id"]: Path(row["video"])
            for row in csv.DictReader(handle)
            if row.get("status", "ok") == "ok" and row.get("video")
        }


def frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()
    return count


def stage_media(source: Path, out_dir: Path, kind: str, sample_id: str) -> str:
    staged = out_dir / "media" / kind / f"{sample_id}{source.suffix}"
    staged.parent.mkdir(parents=True, exist_ok=True)
    if staged.exists() or staged.is_symlink():
        staged.unlink()
    try:
        os.symlink(source, staged)
    except OSError:
        shutil.copy2(source, staged)
    return staged.relative_to(out_dir).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--train_reference_report", type=Path, required=True)
    parser.add_argument("--train_target_report", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--eval_reference_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--training_output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--min_pairs", type=int, default=24)
    parser.add_argument("--validation_count", type=int, default=2)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_references = read_report(args.train_reference_report)
    train_targets = read_report(args.train_target_report)
    pairs = []
    for row in read_jsonl(args.train_manifest):
        sample_id = row["sample_id"]
        reference = train_references.get(sample_id)
        target = train_targets.get(sample_id)
        if reference is None or target is None:
            continue
        reference_frames, target_frames = frame_count(reference), frame_count(target)
        if min(reference_frames, target_frames) < args.frames:
            continue
        staged_target = stage_media(target, args.out_dir, "target", sample_id)
        staged_reference = stage_media(reference, args.out_dir, "reference", sample_id)
        pairs.append(
            {
                "caption": args.prompt,
                "media_path": staged_target,
                "reference_path": staged_reference,
                "sample_id": sample_id,
                "available_frames": min(reference_frames, target_frames),
            }
        )
    if len(pairs) < args.min_pairs:
        raise RuntimeError(f"only {len(pairs)} pairs cover {args.frames} frames; require {args.min_pairs}")
    dataset_path = args.out_dir / "dataset.json"
    dataset_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")

    eval_references = read_report(args.eval_reference_report)
    validation = []
    for row in read_jsonl(args.eval_manifest):
        reference = eval_references.get(row["sample_id"])
        if reference is None or frame_count(reference) < args.frames:
            continue
        staged_reference = stage_media(reference, args.out_dir, "validation_reference", row["sample_id"])
        validation.append(
            {
                "sample_id": row["sample_id"],
                "prompt": args.prompt,
                "reference_path": str(args.out_dir / staged_reference),
            }
        )
        if len(validation) >= args.validation_count:
            break
    if len(validation) != args.validation_count:
        raise RuntimeError(f"found {len(validation)}/{args.validation_count} validation references")
    (args.out_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    config = {
        "model": {"model_source": "LTXV_2B_0.9.6_DEV", "training_mode": "lora", "load_checkpoint": None},
        "lora": {
            "rank": args.rank,
            "alpha": args.rank,
            "dropout": 0.0,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        },
        "conditioning": {"mode": "reference_video", "first_frame_conditioning_p": 0.0, "reference_latents_dir": "reference_latents"},
        "optimization": {
            "learning_rate": 1e-4,
            "steps": args.steps,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 1.0,
            "optimizer_type": "adamw8bit",
            "scheduler_type": "linear",
            "scheduler_params": {},
            "enable_gradient_checkpointing": True,
        },
        "acceleration": {
            "mixed_precision_mode": "bf16",
            # The server image has the CUDA runtime but no nvcc, so Quanto's
            # FP8 Marlin kernels cannot be JIT-compiled. The 2B model fits the
            # 48 GB smoke GPU in BF16 with gradient checkpointing.
            "quantization": None,
            "load_text_encoder_in_8bit": True,
            "compile_with_inductor": False,
            "compilation_mode": "reduce-overhead",
        },
        "data": {"preprocessed_data_root": str(args.out_dir / ".precomputed"), "num_dataloader_workers": 0},
        "validation": {
            "prompts": [item["prompt"] for item in validation],
            "reference_videos": [item["reference_path"] for item in validation],
            "negative_prompt": "inconsistent motion, changed camera, blurry, jittery, distorted",
            "images": None,
            "video_dims": [args.width, args.height, args.frames],
            "seed": 42,
            "inference_steps": 20,
            "interval": args.checkpoint_interval,
            "videos_per_prompt": 1,
            "guidance_scale": 3.0,
            "skip_initial_validation": False,
        },
        "checkpoints": {"interval": args.checkpoint_interval, "keep_last_n": 3},
        "flow_matching": {"timestep_sampling_mode": "shifted_logit_normal", "timestep_sampling_params": {}},
        "hub": {"push_to_hub": False, "hub_model_id": None},
        "wandb": {"enabled": False, "project": "ltxv-trainer", "entity": None, "tags": ["aicity-track5", "ic-lora-smoke"], "log_validation_videos": False},
        "seed": 42,
        "output_dir": str(args.training_output),
    }
    args.config_path.parent.mkdir(parents=True, exist_ok=True)
    args.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"pairs={len(pairs)}")
    print(f"dataset={dataset_path}")
    print(f"validation={args.out_dir / 'validation.json'}")
    print(f"config={args.config_path}")


if __name__ == "__main__":
    main()
