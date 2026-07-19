#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft
ENV_DIR="$ROOT/cosmos25_diffusers_env"
CACHE_DIR="$ROOT/framepack_hf_cache"
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
OUT="$ROOT/framepack_f1_bdd4_gate"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== FramePack F1 BDD4 validation gate ==="
echo "manifest=$MANIFEST"
echo "out=$OUT"
echo "gpu=$GPU_ID"

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 42000
mkdir -p "$OUT" "$CACHE_DIR"

cat > "$OUT/run_framepack_f1_bdd4.py" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
from diffusers.utils import export_to_video
from PIL import Image
from transformers import SiglipImageProcessor, SiglipVisionModel


ROOT = Path("/workspace/experiments/wan_ticft")
MANIFEST = ROOT / "ltxv_bdd25_gate/dense40_manifest.jsonl"
OUT = ROOT / "framepack_f1_bdd4_gate"
VIDEO_DIR = OUT / "videos"
INPUT_DIR = OUT / "inputs"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
with MANIFEST.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("view_type") == "bdd":
            rows.append(row)
        if len(rows) >= 4:
            break

prompt_suffix = (
    " Continue the exact same forward-facing driving camera scene. Preserve road layout, "
    "dashboard, lighting, camera viewpoint, vehicles, pedestrians, signs, and object identities. "
    "Only realistic near-future traffic motion should change."
)

transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
    "lllyasviel/FramePack_F1_I2V_HY_20250503",
    torch_dtype=torch.bfloat16,
)
feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="feature_extractor")
image_encoder = SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl",
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
pipe = HunyuanVideoFramepackPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

report = []
for pos, row in enumerate(rows, start=1):
    sample_id = row["sample_id"]
    cond_paths = (row.get("frames") or {}).get("condition") or []
    img = Image.open(cond_paths[-1]).convert("RGB")
    input_path = INPUT_DIR / f"{sample_id}.png"
    img.save(input_path)
    prompt = (row.get("prompt") or "Traffic video prediction.")[:900] + prompt_suffix
    frames = pipe(
        image=img,
        prompt=prompt,
        height=512,
        width=768,
        num_frames=17,
        num_inference_steps=8,
        guidance_scale=9.0,
        generator=torch.Generator().manual_seed(1000 + pos),
        sampling_type="vanilla",
    ).frames[0]
    out_video = VIDEO_DIR / f"{sample_id}.mp4"
    export_to_video(frames, str(out_video), fps=16)
    report.append({"sample_id": sample_id, "video": str(out_video), "status": "ok", "future_frames": len(frames)})
    print(f"[{pos}/{len(rows)}] {sample_id} frames={len(frames)} out={out_video}", flush=True)

with (OUT / "report.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "future_frames"])
    writer.writeheader()
    writer.writerows(report)
with (OUT / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

START=$(date +%s)
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONPATH=/workspace \
HF_HUB_CACHE="$CACHE_DIR" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
timeout 180m "$ENV_DIR/bin/python" "$OUT/run_framepack_f1_bdd4.py"
ELAPSED=$(( $(date +%s) - START ))
echo "elapsed_seconds=$ELAPSED"

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    framepack_f1="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report framepack_f1="$OUT/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 2

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$OUT/videos" \
  --mode videos \
  --out_dir "${OUT}_sheets" \
  --samples 12 \
  --tile_width 320 \
  --cols 4

echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_sheets"
