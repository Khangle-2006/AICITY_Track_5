from __future__ import annotations

import argparse
import csv
import json
import types
from pathlib import Path

import cv2
import torch
from diffusers import DiffusionPipeline
from diffusers_cosmos3.pipeline import save_img_or_video

import diffusers_cosmos3  # noqa: F401 - registers Cosmos3 classes in diffusers


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_anchor(cv2m, row: dict, path: Path, width: int, height: int) -> None:
    image_path = Path((row.get("frames") or {}).get("condition", [])[-1])
    image = cv2m.imread(str(image_path), cv2m.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {image_path}")
    image = cv2m.resize(image, (width, height), interpolation=cv2m.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2m.imwrite(str(path), image):
        raise RuntimeError(f"cannot write {path}")


def prompt_for(row: dict) -> str:
    base = " ".join(str(row.get("prompt", "")).split()[:90])
    return " ".join(
        [
            "Use the reference image as a strict first frame.",
            "Continue the same forward-facing dashcam traffic scene.",
            "Preserve exact road layout, lanes, buildings, vehicles, pedestrians, windshield, dashboard, camera angle, horizon, and lighting.",
            "Only advance existing objects slightly in time with realistic motion.",
            "Do not change the location, do not add new vehicles, do not add new pedestrians, do not rotate the camera.",
            base,
        ]
    ).strip()


def patch_steps(pipe, steps: int | None):
    if steps is None:
        return
    original = pipe._resolve_defaults_and_prompts

    def wrapped(self, prompt, negative_prompt, image, num_frames, fps, height, width):
        guidance, _, shift, prompt, negative_prompt = original(prompt, negative_prompt, image, num_frames, fps, height, width)
        return guidance, steps, shift, prompt, negative_prompt

    pipe._resolve_defaults_and_prompts = types.MethodType(wrapped, pipe)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cosmos3-Nano diffusers image-anchor smoke on BDD rows.")
    parser.add_argument("--manifest", type=Path, default=Path("/workspace/experiments/wan_ticft/ltxv_bdd25_gate/selected_manifest.jsonl"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", default="/root/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/fea6e03ac3d7884b4105ed8ee79fc480fca70965")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=300)
    args = parser.parse_args()

    cv2m = cv2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"][: args.limit]
    if not rows:
        raise RuntimeError("no rows selected")

    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.to("cuda")
    patch_steps(pipe, args.steps)

    report = []
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for index, row in enumerate(rows, 1):
        sid = row["sample_id"]
        anchor = args.out_dir / "anchor_images" / f"{sid}.png"
        write_anchor(cv2m, row, anchor, args.width, args.height)
        generator = torch.Generator(device="cuda").manual_seed(args.seed + index - 1)
        print(f"[{index}/{len(rows)}] {sid} | generating", flush=True)
        result = pipe(
            prompt=prompt_for(row),
            negative_prompt="new scene, changed road, changed buildings, changed vehicles, changed pedestrians, rotated camera, tilted horizon, blur, ghosting, double exposure, warped geometry, hallucinated objects, low quality, jittery, deformed",
            image=str(anchor),
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
            condition_frame_indexes=[0],
            generator=generator,
            use_system_prompt=True,
            device="cuda",
            dtype=torch.bfloat16,
            output_type="video",
        )[0]
        out_stem = args.out_dir / "videos" / sid
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        save_img_or_video(result, str(out_stem), fps=args.fps)
        out_video = out_stem.with_suffix(".mp4")
        report.append({"sample_id": sid, "video": str(out_video), "status": "ok", "source_fps": f"{args.fps:.6f}", "future_frames": args.num_frames})
        print(f"[{index}/{len(rows)}] {sid} | {out_video}", flush=True)

    with (args.out_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "source_fps", "future_frames"])
        writer.writeheader()
        writer.writerows(report)
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(report)}, indent=2))


if __name__ == "__main__":
    main()
