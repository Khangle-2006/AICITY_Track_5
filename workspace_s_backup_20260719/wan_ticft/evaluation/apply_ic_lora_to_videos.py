"""Faithful IC-LoRA apply: load DEV LTX + trained IC-LoRA, refine each input video using the
video ITSELF as the reference_video (synthetic->real detailer). Preserves layout; outputs refined
videos for REAL-FID gating. Chunks long videos to the model's frame budget."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import torch
from diffusers.utils import export_to_video
from baseline.frame_io import import_cv2_np
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
from ltxv_trainer.model_loader import load_ltxv_components


def read_video(cv2, path, w, h):
    cap = cv2.VideoCapture(str(path)); frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return frames, fps


def to_tensor(np_mod, frames):
    import numpy as np
    arr = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in_dir", type=Path, required=True, help="dir of input videos (reference=these)")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--chunk", type=int, default=9)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--prompt", default="Restore this traffic prediction video while preserving exact camera geometry and actor motion.")
    a = p.parse_args()
    cv2, np = import_cv2_np()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    comp = load_ltxv_components(model_source="LTXV_2B_0.9.6_DEV", load_text_encoder_in_8bit=False,
                               transformer_dtype=torch.bfloat16, vae_dtype=torch.bfloat16)
    pipe = LTXConditionPipeline(scheduler=comp.scheduler, vae=comp.vae, text_encoder=comp.text_encoder,
                               tokenizer=comp.tokenizer, transformer=comp.transformer).to("cuda")
    pipe.load_lora_weights(str(a.checkpoint.parent), weight_name=a.checkpoint.name)
    pipe.set_progress_bar_config(disable=True)

    vids = [v for v in sorted(a.in_dir.glob("*")) if v.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
    if a.limit:
        vids = vids[: a.limit]
    rows = []
    for i, v in enumerate(vids, 1):
        frames, fps = read_video(cv2, v, a.width, a.height)
        refined = []
        for s in range(0, len(frames), a.chunk):
            seg = frames[s : s + a.chunk]
            if len(seg) < 2:
                refined.extend(seg); break
            ref = to_tensor(np, seg)
            g = torch.Generator(device="cuda").manual_seed(a.seed + s)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = pipe(reference_video=ref, output_reference_comparison=False, prompt=a.prompt,
                           negative_prompt="synthetic texture, cartoon, blurry, distorted, inconsistent geometry",
                           width=a.width, height=a.height, num_frames=len(seg), frame_rate=int(round(fps)),
                           num_inference_steps=a.steps, guidance_scale=a.guidance, generator=g)
            refined.extend(list(out.frames[0])[: len(seg)])
        op = a.out_dir / f"{v.stem}.mp4"
        export_to_video(refined[: len(frames)], str(op), fps=int(round(fps)))
        rows.append({"sample_id": v.stem, "video": str(op), "status": "ok"})
        print(f"[{i}/{len(vids)}] {v.stem} | {len(frames)} frames", flush=True)
    with (a.out_dir / "report.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["sample_id", "video", "status"]); w.writeheader(); w.writerows(rows)
    print(f"output={a.out_dir}")


if __name__ == "__main__":
    main()
