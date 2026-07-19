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


def prompt_for(row: dict) -> str:
    base = " ".join(str(row.get("prompt", "")).split()[:90])
    return " ".join(
        [
            "Refine the conditioned dashcam video while preserving the exact scene geometry.",
            "Keep the same road layout, lanes, buildings, vehicles, pedestrians, traffic lights, windshield, dashboard, camera angle, horizon, weather, and lighting.",
            "Remove ghosting, double exposure, smeared road markings, and synthetic blur.",
            "Do not add or remove vehicles or pedestrians. Do not change camera pose.",
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


def read_video_tensor(path: Path, width: int, height: int, num_frames: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1))
    cap.release()
    if not frames:
        raise RuntimeError(f"cannot read scaffold video {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())
    video = torch.stack(frames, dim=1).float() / 127.5 - 1.0
    return video.contiguous()


def condition_indexes(mode: str, num_frames: int) -> list[int]:
    if mode == "first":
        return [0]
    if mode == "thirds":
        return sorted(set([0, num_frames // 2, num_frames - 1]))
    if mode == "even":
        return list(range(0, num_frames, 2))
    if mode == "all":
        return list(range(num_frames))
    raise ValueError(f"unknown condition mode {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cosmos3-Nano diffusers with a conditioned scaffold video.")
    parser.add_argument("--manifest", type=Path, default=Path("/workspace/experiments/wan_ticft/ltxv_bdd25_gate/selected_manifest.jsonl"))
    parser.add_argument("--scaffold_dir", type=Path, default=Path("/workspace/experiments/wan_ticft/ltx_detail_donor_bdd24_gate/ltxhf_t55/videos"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", default="/root/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/fea6e03ac3d7884b4105ed8ee79fc480fca70965")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--condition_mode", choices=["first", "thirds", "even", "all"], default="thirds")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in read_jsonl(args.manifest):
        if row.get("view_type") != "bdd":
            continue
        sid = row["sample_id"]
        if (args.scaffold_dir / f"{sid}.mp4").exists():
            rows.append(row)
        if len(rows) >= args.limit:
            break
    if not rows:
        raise RuntimeError("no rows with scaffold videos selected")

    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.to("cuda")
    patch_steps(pipe, args.steps)
    cond_idx = condition_indexes(args.condition_mode, args.num_frames)

    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = []
    for index, row in enumerate(rows, 1):
        sid = row["sample_id"]
        scaffold_path = args.scaffold_dir / f"{sid}.mp4"
        scaffold = read_video_tensor(scaffold_path, args.width, args.height, args.num_frames)
        generator = torch.Generator(device="cuda").manual_seed(args.seed + index - 1)
        print(f"[{index}/{len(rows)}] {sid} | generating cond={cond_idx}", flush=True)
        result = pipe(
            prompt=prompt_for(row),
            negative_prompt="changed road, changed buildings, new vehicles, new pedestrians, duplicated cars, duplicated pedestrians, rotated camera, tilted horizon, blur, ghosting, double exposure, warped geometry, smeared lane markings, light trails, hallucinated objects, low quality, jittery, deformed",
            image=scaffold,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
            condition_frame_indexes=cond_idx,
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
        report.append(
            {
                "sample_id": sid,
                "video": str(out_video),
                "scaffold": str(scaffold_path),
                "status": "ok",
                "source_fps": f"{args.fps:.6f}",
                "future_frames": args.num_frames,
                "condition_mode": args.condition_mode,
            }
        )
        print(f"[{index}/{len(rows)}] {sid} | {out_video}", flush=True)

    with (args.out_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "video", "scaffold", "status", "source_fps", "future_frames", "condition_mode"],
        )
        writer.writeheader()
        writer.writerows(report)
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(report), "condition_indexes": cond_idx}, indent=2))


if __name__ == "__main__":
    main()
