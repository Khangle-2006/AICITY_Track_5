#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/LTX-Video-Trainer
ENV=/workspace/experiments/wan_ticft/ltx_lora_trainer_env
OUT=/workspace/experiments/wan_ticft/ltx_lora_bdd_smoke_eval4
CHECKPOINT=/workspace/experiments/wan_ticft/ltx_lora_bdd_smoke20_output/checkpoints/lora_weights_step_00050.safetensors
MANIFEST=/workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"

echo "=== LTX LoRA 4-sample quality gate ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 20000
test -f "$CHECKPOINT"

rm -rf "$OUT"
PYTHONPATH="/workspace:$REPO/src" TOKENIZERS_PARALLELISM=false \
"$ENV/bin/python" -m wan_ticft.evaluation.evaluate_ltx_lora_smoke \
  --manifest "$MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --out_dir "$OUT" \
  --limit 4

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    base="$OUT/base_report.csv" \
    lora="$OUT/lora_report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --limit 4 \
  --out_csv "$OUT/eval.csv"

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$OUT/lora" \
  --mode videos \
  --out_dir "${OUT}_lora_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

echo "Inspect: ${OUT}_lora_sheets"
echo "Gate rule: continue only if LoRA clearly improves base without visible degradation."
