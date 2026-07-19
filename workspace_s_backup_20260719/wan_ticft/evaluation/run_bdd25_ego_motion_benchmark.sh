#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python -m wan_ticft.evaluation.evaluate_bdd_ego_motion \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --out_csv /workspace/experiments/wan_ticft/bdd25_ego_motion_baselines.csv
