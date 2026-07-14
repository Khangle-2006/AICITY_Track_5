#!/bin/bash
# run_pipeline.sh

# Run generation
CUDA_VISIBLE_DEVICES=2 python3 wts_baseline/scripts/infer/infer_wan2_2.py \
    --manifest wts_baseline/data/manifests/test.jsonl \
    --output_dir wts_baseline/prediction_wan2.2_ticft_sr \
    --prompt_embeds_dir /tmp/prompt_embeds \
    --max_samples 100 \
    --num_inference_steps 50 \
    --vae_dtype float16 --lambda_weight 0.0 \
    --use_ticft --use_sr --h 16 \
    --enable_sequential_cpu_offload

# Compile folders to MP4s
for scenario_dir in wts_baseline/prediction_wan2.2_ticft_sr/*/; do
  scenario=$(basename "$scenario_dir")
  if [[ "$scenario" != *"_gt" && "$scenario" != "sa2va_understanding" && -d "$scenario_dir" ]]; then
    if [ ! -f "wts_baseline/prediction_wan2.2_ticft_sr/$scenario.mp4" ]; then
      ffmpeg -y -framerate 10 -i "wts_baseline/prediction_wan2.2_ticft_sr/$scenario/%d.png" \
        -c:v libx264 -pix_fmt yuv420p \
        "wts_baseline/prediction_wan2.2_ticft_sr/$scenario.mp4"
    fi
  fi
done
