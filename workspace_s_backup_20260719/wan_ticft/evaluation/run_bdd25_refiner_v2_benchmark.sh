#!/usr/bin/env bash
set -euo pipefail

cd /workspace

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_bdd_ego_motion \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --refiner_checkpoint /workspace/experiments/refiner_bdd_direct_h1_120_v2/refiner_best.pt \
  --out_csv /workspace/experiments/wan_ticft/bdd25_refiner_v2_benchmark.csv
