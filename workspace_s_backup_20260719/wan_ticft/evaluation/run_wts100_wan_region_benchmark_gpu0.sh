#!/usr/bin/env bash
set -euo pipefail

cd /workspace

MANIFEST=/workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl
WAN_OUT=/workspace/experiments/wan_ticft/wan22_wts100_s10
MOTION_CSV=/workspace/experiments/wan_ticft/all125_sparse_target_motion_baselines.csv

python -m wan_ticft.inference.run_wan22_ti2v_parallel \
  --manifest "$MANIFEST" \
  --out_dir "$WAN_OUT" \
  --gpus 0 \
  --offset 25 \
  --limit 100 \
  --frame_num 121 \
  --sample_steps 10 \
  --base_seed 11 \
  --skip_existing

python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports wan="$WAN_OUT/report.csv" \
  --offset 25 \
  --limit 100 \
  --offset_mode frame \
  --out_csv /workspace/experiments/wan_ticft/wan22_wts100_eval_frame.csv

python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports wan="$WAN_OUT/report.csv" \
  --offset 25 \
  --limit 100 \
  --offset_mode timestamp \
  --out_csv /workspace/experiments/wan_ticft/wan22_wts100_eval_timestamp.csv

python -m wan_ticft.evaluation.analyze_wan_region_router \
  --wan_eval_csv /workspace/experiments/wan_ticft/wan22_wts100_eval_frame.csv \
  --motion_csv "$MOTION_CSV" \
  --out_prefix /workspace/experiments/wan_ticft/wan_region_router_frame

python -m wan_ticft.evaluation.analyze_wan_region_router \
  --wan_eval_csv /workspace/experiments/wan_ticft/wan22_wts100_eval_timestamp.csv \
  --motion_csv "$MOTION_CSV" \
  --out_prefix /workspace/experiments/wan_ticft/wan_region_router_timestamp
