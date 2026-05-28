#!/usr/bin/env bash
set -euo pipefail

# Parallel execution on 4 GPUs for Wan2.2 MoE Video-to-Video SOTA Inference

export PYTHONPATH=/workspace/wts_baseline:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST="${1:-data/manifests/test.jsonl}"
OUTPUT_DIR="${2:-prediction_wan2.2_sota}"
MODEL_ID="${3:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
MAX_SAMPLES="${4:-0}"  # 0 means all samples

EXTRA_ARGS=""
if [ "$MAX_SAMPLES" -gt 0 ]; then
  EXTRA_ARGS="--max_samples $MAX_SAMPLES"
fi

WORLD_SIZE=4

echo "Launching parallel inference across ${WORLD_SIZE} GPUs..."

# Launch processes for Rank 0 to 3
for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
  GPU_ID=$RANK
  echo "Starting rank $RANK on GPU $GPU_ID..."
  
  CUDA_VISIBLE_DEVICES=$GPU_ID python3 -u scripts/infer/infer_wan2_2.py \
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
    --rank "$RANK" \
    --world_size "$WORLD_SIZE" \
    $EXTRA_ARGS > "rank_${RANK}.log" 2>&1 &
done

echo "All ranks launched. You can monitor progress with 'tail -f rank_*.log'."
echo "Waiting for all processes to finish..."
wait
echo "Parallel inference completed successfully!"
