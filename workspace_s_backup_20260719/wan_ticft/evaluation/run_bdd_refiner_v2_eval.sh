#!/usr/bin/env bash
set -euo pipefail

cd /workspace

CHECKPOINT=/workspace/experiments/refiner_bdd_direct_h1_120_v2/refiner_best.pt

for HORIZON in 8 30 60 120; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python -m baseline.eval_refiner \
    --index_dir /workspace/experiments/index \
    --checkpoint "$CHECKPOINT" \
    --view_type bdd \
    --max_rows 0 \
    --target_gap_frames "$HORIZON" \
    --eval_horizon "$HORIZON" \
    --batch_size 16 \
    --num_workers 4
done

