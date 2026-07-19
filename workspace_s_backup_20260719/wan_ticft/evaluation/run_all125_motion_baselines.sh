#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python -m wan_ticft.evaluation.evaluate_manifest_motion_baselines \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --source_future_frames 120 \
  --out_csv /workspace/experiments/wan_ticft/all125_121f_motion_baselines_ffmpeg.csv
