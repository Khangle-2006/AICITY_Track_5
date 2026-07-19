#!/usr/bin/env bash
set -euo pipefail

cd /workspace

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m baseline.train_refiner \
  --index_dir /workspace/experiments/index \
  --out_dir /workspace/experiments/refiner_bdd_direct_h1_120_v2_smoke \
  --view_type bdd \
  --max_train_rows 64 \
  --max_val_rows 16 \
  --num_history 8 \
  --target_gap_frames 60 \
  --eval_horizon 60 \
  --random_horizon \
  --horizon_min 1 \
  --horizon_max 120 \
  --time_normalized_features \
  --include_context_features \
  --train_width 320 \
  --train_height 180 \
  --batch_size 8 \
  --epochs 1 \
  --lr 3e-4 \
  --num_workers 2 \
  --model_width 64 \
  --residual_scale 0.35 \
  --lambda_grad 0.05 \
  --lambda_residual 0.03 \
  --max_steps_per_epoch 4 \
  --max_eval_batches 2

