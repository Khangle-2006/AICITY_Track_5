#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft
ENV_DIR="$ROOT/cosmos25_diffusers_env"
CACHE_DIR="$ROOT/skyreels_v2_hf_cache"
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
OUT="$ROOT/skyreels_v2_i2v_bdd1_smoke"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== SkyReels-V2 I2V 1.3B BDD1 smoke ==="
echo "manifest=$MANIFEST"
echo "out=$OUT"
echo "gpu=$GPU_ID"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing env: $ENV_DIR" >&2
  exit 2
fi
if ! "$ENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import ftfy
PY
then
  "$ENV_DIR/bin/python" -m pip install ftfy
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 2
fi

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 42000

FREE_GIB=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( FREE_GIB < 45 )); then
  echo "Need at least 45 GiB free on /workspace; found ${FREE_GIB} GiB." >&2
  exit 2
fi

mkdir -p "$OUT" "$CACHE_DIR"

cat > "$OUT/run_skyreels_v2_i2v_bdd1.py" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, SkyReelsV2ImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image


ROOT = Path("/workspace/experiments/wan_ticft")
MANIFEST = ROOT / "ltxv_bdd25_gate/dense40_manifest.jsonl"
OUT = ROOT / "skyreels_v2_i2v_bdd1_smoke"
VIDEO_DIR = OUT / "videos"
INPUT_DIR = OUT / "inputs"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

row = None
with MANIFEST.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("view_type") == "bdd":
            row = candidate
            break
if row is None:
    raise RuntimeError("No BDD row found in manifest")

sample_id = row["sample_id"]
cond_paths = (row.get("frames") or {}).get("condition") or []
if not cond_paths:
    raise RuntimeError(f"{sample_id}: missing condition frame paths")
image = load_image(cond_paths[-1]).convert("RGB")
input_path = INPUT_DIR / f"{sample_id}_last_condition.png"
image.save(input_path)

prompt = (
    "Forward-facing dashcam video from a moving car on a city street. "
    "The car continues driving forward smoothly. Preserve the same camera viewpoint, road, lane markings, "
    "buildings, parked bikes, vehicles, lighting, shadows, dashboard, and object identities. "
    "Only realistic short-term driving motion changes."
)

model_id = "Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers"
print(f"loading {model_id}", flush=True)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = SkyReelsV2ImageToVideoPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
pipe.to("cuda")
try:
    pipe.vae.enable_tiling()
except Exception:
    pass

print(f"running sample={sample_id}", flush=True)
frames = pipe(
    image=image,
    prompt=prompt,
    height=544,
    width=960,
    num_frames=97,
    num_inference_steps=12,
    guidance_scale=5.0,
    generator=torch.Generator(device="cuda").manual_seed(20260621),
).frames[0]

out_video = VIDEO_DIR / f"{sample_id}.mp4"
export_to_video(frames, str(out_video), fps=16, quality=8)
print(f"output={out_video} frames={len(frames)}", flush=True)

with (OUT / "report.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "future_frames"])
    writer.writeheader()
    writer.writerow({"sample_id": sample_id, "video": str(out_video), "status": "ok", "future_frames": len(frames)})
with (OUT / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

START=$(date +%s)
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONPATH=/workspace \
HF_HUB_CACHE="$CACHE_DIR" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
timeout 180m "$ENV_DIR/bin/python" "$OUT/run_skyreels_v2_i2v_bdd1.py"
ELAPSED=$(( $(date +%s) - START ))
echo "elapsed_seconds=$ELAPSED"

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    skyreels_v2_i2v="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report skyreels_v2_i2v="$OUT/report.csv" \
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

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    skyreels_v2_i2v="$OUT/report.csv" \
  --sample_id val_bdd_pc_5k_video7040_phase0 \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 220

echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_sheets"
echo "compare_sheets=${OUT}_compare_sheets"
