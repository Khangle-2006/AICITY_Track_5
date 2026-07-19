#!/usr/bin/env bash
set -euo pipefail

cd /workspace

MANIFEST=/workspace/experiments/wan_ticft/manifests/val_ticft_vehicle_only.jsonl
WAN_REPORT=/workspace/experiments/wan_ticft/wan22_ti2v_parallel_vehicle4/report.csv
MOTION_CSV=/workspace/experiments/wan_ticft/all125_sparse_target_motion_baselines.csv

python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports wan="$WAN_REPORT" \
  --limit 4 \
  --offset_mode frame \
  --out_csv /workspace/experiments/wan_ticft/wan22_vehicle4_eval_frame.csv

python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports wan="$WAN_REPORT" \
  --limit 4 \
  --offset_mode timestamp \
  --out_csv /workspace/experiments/wan_ticft/wan22_vehicle4_eval_timestamp.csv

python -m wan_ticft.evaluation.analyze_wan_region_router \
  --wan_eval_csv /workspace/experiments/wan_ticft/wan22_vehicle4_eval_frame.csv \
  --motion_csv "$MOTION_CSV" \
  --out_prefix /workspace/experiments/wan_ticft/wan_region_router_vehicle4_frame

python -m wan_ticft.evaluation.analyze_wan_region_router \
  --wan_eval_csv /workspace/experiments/wan_ticft/wan22_vehicle4_eval_timestamp.csv \
  --motion_csv "$MOTION_CSV" \
  --out_prefix /workspace/experiments/wan_ticft/wan_region_router_vehicle4_timestamp
