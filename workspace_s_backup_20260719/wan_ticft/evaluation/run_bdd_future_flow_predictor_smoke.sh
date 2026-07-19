#!/usr/bin/env bash
set -euo pipefail

cd /workspace

GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"
OUT=/workspace/experiments/wan_ticft/bdd_future_flow_predictor_smoke

echo "=== BDD history-to-future flow predictor smoke ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 6000

rm -rf "$OUT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.train_bdd_future_flow_predictor \
  --index_dir /workspace/experiments/index \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --out_dir "$OUT" \
  --max_train_rows 1000 \
  --width 320 \
  --height 180 \
  --max_horizon 45 \
  --epochs 4 \
  --batch_size 8 \
  --num_workers 4 \
  --model_width 48

echo
echo "Gate rule: continue only if predicted_future_flow clearly beats repeat_last on BDD25."
