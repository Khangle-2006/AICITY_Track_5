#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/LTX-Video-Trainer
ENV=/workspace/experiments/wan_ticft/ltx_lora_trainer_env
OUT=/workspace/experiments/wan_ticft/ltx_ic_lora_bdd_smoke_eval5
CHECKPOINT=/workspace/experiments/wan_ticft/ltx_ic_lora_bdd_smoke20_output/checkpoints/lora_weights_step_00050.safetensors
MANIFEST=/workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== LTX IC-LoRA balanced 5-sample quality gate ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 20000
test -f "$CHECKPOINT"

rm -rf "$OUT"
PYTHONPATH="/workspace:$REPO/src" TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$ENV/bin/python" -m wan_ticft.evaluation.evaluate_ltx_ic_lora_smoke \
  --manifest "$MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --out_dir "$OUT" \
  --per_phase 1

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    base_reference="$OUT/base_report.csv" \
    ic_lora="$OUT/ic_lora_report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "$OUT/eval.csv"

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$OUT/ic_lora" \
  --mode videos \
  --out_dir "${OUT}_ic_lora_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

echo "Inspect: ${OUT}_ic_lora_sheets"
echo "Gate rule: expand only if IC-LoRA beats base_reference and has no visible degradation."
