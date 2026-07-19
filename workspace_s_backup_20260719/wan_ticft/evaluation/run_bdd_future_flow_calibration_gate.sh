#!/usr/bin/env bash
set -euo pipefail

cd /workspace

GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"
CHECKPOINT=/workspace/experiments/wan_ticft/bdd_future_flow_predictor_smoke/future_flow_predictor.pt
OUT=/workspace/experiments/wan_ticft/bdd_future_flow_calibration_gate

echo "=== BDD predicted future-flow amplitude calibration ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
test -f "$CHECKPOINT"

rm -rf "$OUT"
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.train_bdd_future_flow_predictor \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --out_dir "$OUT" \
  --checkpoint "$CHECKPOINT" \
  --width 320 \
  --height 180 \
  --max_horizon 45 \
  --model_width 48 \
  --eval_flow_scales 0.25 0.5 0.75 1.0

echo
echo "Gate rule: abandon direct dense-flow prediction unless a calibrated scale clearly beats repeat_last."
