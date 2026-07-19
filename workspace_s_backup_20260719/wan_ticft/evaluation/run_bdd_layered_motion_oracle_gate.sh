#!/usr/bin/env bash
set -euo pipefail

cd /workspace

echo "=== BDD layered global + residual-flow oracle gate ==="
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_bdd_layered_motion_oracle \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --out_csv /workspace/experiments/wan_ticft/bdd_layered_motion_oracle_gate.csv \
  --estimate_width 640 \
  --roi_bottom 0.85

echo
echo "Gate rule:"
echo "  - Continue with a layered predictor only if oracle_layered_flow retains most of direct-flow oracle quality."
echo "  - Residual/direct flow ratio should be materially below 1.0, otherwise decomposition did not simplify learning."
