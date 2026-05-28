#!/usr/bin/env bash
set -euo pipefail

# Wan2.2 MoE video-to-video inference helper script

export PYTHONPATH=/workspace/wts_baseline:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_ID="${2:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
MANIFEST="${3:-data/manifests/test.jsonl}"
OUTPUT_DIR="${4:-prediction_wan2.2_sota}"
MAX_SAMPLES="${5:-1}"

EXTRA_ARGS=""
if [ "$MAX_SAMPLES" -gt 0 ]; then
  EXTRA_ARGS="--max_samples $MAX_SAMPLES"
fi

python3 -u scripts/infer/infer_wan2_2.py \
  --manifest "$MANIFEST" \
  --output_dir "$OUTPUT_DIR" \
  --model_id "$MODEL_ID" \
  --torch_dtype bfloat16 \
  --height 480 \
  --width 832 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --min_guidance 1.5 \
  --strength 0.7 \
  --flow_shift 3.0 \
  --chunk_size 81 \
  --h 16 \
  --overlap_frames 2 \
  --output_width 1280 \
  --output_height 720 \
  --use_background_preservation \
  --use_alpha_compositing \
  --use_dynamic_cfg \
  --lambda_weight 0.1 \
  --num_workers 2 \
  $EXTRA_ARGS
