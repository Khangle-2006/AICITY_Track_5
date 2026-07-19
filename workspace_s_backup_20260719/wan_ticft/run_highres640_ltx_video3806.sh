#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/workspace

OUT_DIR=/workspace/experiments/wan_ticft/ltx_highres640_motionprompt_video3806_20260708
mkdir -p "$OUT_DIR"

PROMPT='Continue the same real dashcam video from the final observed frame. The ego vehicle keeps driving forward along the lane. All visible cars on the same road and neighboring lanes continue moving smoothly forward with traffic, changing size and position consistently; do not freeze nearby cars as background. Keep buildings, curb, lane markings, sidewalk, sky, camera viewpoint, exposure, windshield reflections, compression texture, and static background geometry stable. Preserve object identities and realistic road scale. No double exposure, no ghost trails, no new cars, no cinematic render, no beautification.'

/opt/conda/envs/wan22/bin/python /workspace/wan_ticft/run_ltxv_seed_video3806.py \
  --manifest /workspace/experiments/wan_ticft/memory_scaffold_ltx_video3806_s085/manifest_video3806.jsonl \
  --out_dir "$OUT_DIR" \
  --seeds 17 \
  --prompt_override "$PROMPT" \
  --max_history_frames 17 \
  --height 640 \
  --width 1152 \
  --num_frames 81 \
  --future_frames 59 \
  --image_cond_noise_scale 0.025 \
  --vae_tiling 1 \
  --resume
