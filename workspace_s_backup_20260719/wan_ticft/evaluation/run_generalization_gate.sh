#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft

echo "=== Sparse-target cross-domain generalization ==="
"$PYTHON" -m wan_ticft.evaluation.summarize_generalization_gate \
  --input_csv "$ROOT/all125_sparse_target_motion_baselines.csv" \
  --out_csv "$ROOT/all125_sparse_target_generalization.csv"

echo
echo "=== Long-horizon cross-domain generalization ==="
"$PYTHON" -m wan_ticft.evaluation.summarize_generalization_gate \
  --input_csv "$ROOT/all125_121f_motion_baselines.csv" \
  --out_csv "$ROOT/all125_121f_generalization.csv"

echo
echo "Generalization rule: promote only candidates with Pass=yes on both gates."
