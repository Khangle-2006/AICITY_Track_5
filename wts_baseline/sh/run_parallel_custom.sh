#!/usr/bin/env bash

# Automatically navigate to repository root
cd "$(dirname "$0")/.."
mkdir -p logs

set -euo pipefail

# Custom parallel inference for Wan2.2 MoE Video-to-Video SOTA
# Splits across 2 A100 GPUs with different chunk sizes to respect VRAM limits.

export PYTHONPATH=/workspace/wts_baseline:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST="data/manifests/test.jsonl"
OUTPUT_DIR="prediction_wan2.2_test"
MODEL_ID="Wan-AI/Wan2.2-T2V-A14B-Diffusers"

echo "Starting Rank 0 on GPU 0 with chunk_size 32..."
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/infer/infer_wan2_2.py \
  --manifest "$MANIFEST" \
  --output_dir "$OUTPUT_DIR" \
  --prompt_embeds_dir data/manifests/test_prompt_embeds \
  --model_id "$MODEL_ID" \
  --torch_dtype bfloat16 \
  --height 480 \
  --width 832 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --min_guidance 1.5 \
  --strength 0.7 \
  --flow_shift 3.0 \
  --chunk_size 32 \
  --h 16 \
  --overlap_frames 2 \
  --output_width 1280 \
  --output_height 720 \
  --use_background_preservation \
  --use_alpha_compositing \
  --use_dynamic_cfg \
  --lambda_weight 0.1 \
  --num_workers 2 \
  --rank 0 \
  --world_size 2 \
  --max_samples 0 > logs/rank_0.log 2>&1 &

echo "Starting Rank 1 on GPU 1 with chunk_size 16..."
CUDA_VISIBLE_DEVICES=1 python3 -u scripts/infer/infer_wan2_2.py \
  --manifest "$MANIFEST" \
  --output_dir "$OUTPUT_DIR" \
  --prompt_embeds_dir data/manifests/test_prompt_embeds \
  --model_id "$MODEL_ID" \
  --torch_dtype bfloat16 \
  --height 480 \
  --width 832 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --min_guidance 1.5 \
  --strength 0.7 \
  --flow_shift 3.0 \
  --chunk_size 16 \
  --h 16 \
  --enable_sequential_cpu_offload \
  --overlap_frames 2 \
  --output_width 1280 \
  --output_height 720 \
  --use_background_preservation \
  --use_alpha_compositing \
  --use_dynamic_cfg \
  --lambda_weight 0.1 \
  --num_workers 2 \
  --rank 1 \
  --world_size 2 \
  --max_samples 0 > logs/rank_1.log 2>&1 &

echo "Both ranks launched. Waiting for execution to complete..."
wait
echo "All custom parallel inference completed successfully!"
