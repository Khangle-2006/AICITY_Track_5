#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/LTX-Video-Trainer
ENV=/workspace/experiments/wan_ticft/ltx_lora_trainer_env
DATA=/workspace/experiments/wan_ticft/ltx_lora_bdd_smoke20/.precomputed
CONFIG=/workspace/wan_ticft/evaluation/ltxv_2b_lora_bdd_smoke.yaml
OUT=/workspace/experiments/wan_ticft/ltx_lora_bdd_smoke20_output
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"

echo "=== LTX 2B BDD LoRA 50-step smoke ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 20000

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing trainer environment: $ENV" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

LATENTS="$(find "$DATA/latents" -type f -name '*.pt' | wc -l)"
CONDITIONS="$(find "$DATA/conditions" -type f -name '*.pt' | wc -l)"
echo "precomputed latents=$LATENTS conditions=$CONDITIONS"
if [[ "$LATENTS" -ne 20 || "$CONDITIONS" -ne 20 ]]; then
  echo "Expected exactly 20 latent/condition pairs." >&2
  exit 1
fi

echo
echo "=== Validate config ==="
CONFIG="$CONFIG" PYTHONPATH="$REPO/src" "$ENV/bin/python" - <<'PY'
import os
from pathlib import Path

import yaml
from ltxv_trainer.config import LtxvTrainerConfig

path = Path(os.environ["CONFIG"])
config = LtxvTrainerConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))
print(
    {
        "model": str(config.model.model_source),
        "steps": config.optimization.steps,
        "first_frame_conditioning_p": config.conditioning.first_frame_conditioning_p,
        "quantization": config.acceleration.quantization,
        "output_dir": config.output_dir,
    }
)
if config.optimization.steps != 50 or config.conditioning.first_frame_conditioning_p != 1.0:
    raise SystemExit("smoke gate requires exactly 50 steps and first-frame conditioning probability 1.0")
if config.acceleration.quantization is not None:
    raise SystemExit("smoke gate disables Quanto because this server lacks required CUDA development headers")
PY

echo
echo "=== Train ==="
rm -rf "$OUT"
cd "$REPO"
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$ENV/bin/python" scripts/train.py "$CONFIG"

echo
echo "=== Verify smoke checkpoint ==="
find "$OUT/checkpoints" -maxdepth 1 -type f -name '*step_00050.safetensors' -printf '%f %s bytes\n'
COUNT="$(find "$OUT/checkpoints" -maxdepth 1 -type f -name '*step_00050.safetensors' | wc -l)"
if [[ "$COUNT" -lt 1 ]]; then
  echo "Missing step-50 checkpoint." >&2
  exit 1
fi

echo
echo "=== Training gate result ==="
echo "PASS: 50-step LoRA smoke completed."
echo "Next gate: run LoRA vs unmodified LTX on a very small BDD validation subset."
