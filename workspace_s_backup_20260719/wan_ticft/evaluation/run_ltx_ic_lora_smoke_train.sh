#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/LTX-Video-Trainer
ENV=/workspace/experiments/wan_ticft/ltx_lora_trainer_env
DATA=/workspace/experiments/wan_ticft/ltx_ic_lora_bdd_smoke20/.precomputed
CONFIG=/workspace/wan_ticft/evaluation/ltxv_2b_ic_lora_bdd_smoke.yaml
OUT=/workspace/experiments/wan_ticft/ltx_ic_lora_bdd_smoke20_output
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"

echo "=== LTX 2B BDD IC-LoRA 50-step smoke ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 30000

for NAME in latents conditions reference_latents; do
  COUNT="$(find "$DATA/$NAME" -type f -name '*.pt' | wc -l)"
  echo "$NAME=$COUNT"
  if [[ "$COUNT" -ne 20 ]]; then
    echo "Expected 20 $NAME files." >&2
    exit 1
  fi
done

echo
echo "=== Validate IC-LoRA config ==="
CONFIG="$CONFIG" PYTHONPATH="$REPO/src" "$ENV/bin/python" - <<'PY'
import os
from pathlib import Path
import yaml
from ltxv_trainer.config import LtxvTrainerConfig

config = LtxvTrainerConfig(**yaml.safe_load(Path(os.environ["CONFIG"]).read_text()))
print({
    "mode": config.conditioning.mode,
    "reference_latents_dir": config.conditioning.reference_latents_dir,
    "steps": config.optimization.steps,
    "output_dir": config.output_dir,
})
if config.conditioning.mode != "reference_video" or config.optimization.steps != 50:
    raise SystemExit("invalid IC-LoRA smoke configuration")
PY

echo
echo "=== Train ==="
rm -rf "$OUT"
cd "$REPO"
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$ENV/bin/python" scripts/train.py "$CONFIG"

echo
echo "=== Verify checkpoint ==="
find "$OUT/checkpoints" -maxdepth 1 -type f -name '*step_00050.safetensors' -printf '%f %s bytes\n'
COUNT="$(find "$OUT/checkpoints" -maxdepth 1 -type f -name '*step_00050.safetensors' | wc -l)"
if [[ "$COUNT" -lt 1 ]]; then
  echo "Missing IC-LoRA step-50 checkpoint." >&2
  exit 1
fi
echo "PASS: IC-LoRA 50-step smoke completed."
