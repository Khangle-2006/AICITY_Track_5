#!/usr/bin/env bash

# Automatically navigate to repository root
cd "$(dirname "$0")/.."
mkdir -p logs

set -euo pipefail

# Optimized parallel inference for Wan2.2 MoE Video-to-Video SOTA
# Uses GPU 0 for Rank 0 and GPU 2 for Rank 1
# Uses chunk_size 16 and num_workers 0 to fit in VRAM and avoid CPU RAM thrashing.

export PYTHONPATH=/workspace/wts_baseline:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST="data/manifests/test.jsonl"
OUTPUT_DIR="prediction_wan2.2_test"
MODEL_ID="Wan-AI/Wan2.2-T2V-A14B-Diffusers"

echo "Starting Rank 0 on GPU 0 with chunk_size 16..."
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
  --chunk_size 16 \
  --h 16 \
  --overlap_frames 2 \
  --output_width 1280 \
  --output_height 720 \
  --use_background_preservation \
  --use_alpha_compositing \
  --use_dynamic_cfg \
  --lambda_weight 0.1 \
  --num_workers 0 \
  --rank 0 \
  --world_size 2 \
  --max_samples 0 > logs/rank_0.log 2>&1 &

echo "Starting Rank 1 on GPU 2 with chunk_size 16..."
CUDA_VISIBLE_DEVICES=2 python3 -u scripts/infer/infer_wan2_2.py \
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
  --overlap_frames 2 \
  --output_width 1280 \
  --output_height 720 \
  --use_background_preservation \
  --use_alpha_compositing \
  --use_dynamic_cfg \
  --lambda_weight 0.1 \
  --num_workers 0 \
  --rank 1 \
  --world_size 2 \
  --max_samples 0 > logs/rank_1.log 2>&1 &

echo "Both ranks launched. Waiting for execution to complete..."
wait
echo "All custom parallel inference completed successfully!"
