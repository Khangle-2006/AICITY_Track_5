#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python -m wan_ticft.evaluation.evaluate_manifest_motion_baselines \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --out_csv /workspace/experiments/wan_ticft/all125_sparse_target_motion_baselines.csv
