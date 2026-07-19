#!/usr/bin/env bash
set -euo pipefail

cd /workspace

echo "=== BDD future global camera-transform oracle gate ==="
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_bdd_global_transform_oracle \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --out_csv /workspace/experiments/wan_ticft/bdd_global_transform_oracle_gate.csv \
  --estimate_width 640 \
  --roi_bottom 0.85

echo
echo "Gate rule:"
echo "  - Continue with camera trajectory prediction only if oracle_global_choice clearly beats repeat_last."
echo "  - Compare against future-flow oracle_direction PSNR=25.4517 to measure residual-motion importance."
