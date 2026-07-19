#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
MANIFEST=/workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl
OUT=/workspace/experiments/wan_ticft/bdd_future_flow_oracle_gate.csv

echo "=== BDD future-flow renderer ceiling gate ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_bdd_future_flow_oracle \
  --manifest "$MANIFEST" \
  --out_csv "$OUT" \
  --flow_width 640

echo
echo "Gate interpretation:"
echo "  - oracle_direction measures geometry-only future-flow rendering."
echo "  - oracle_block32 measures a strong ceiling with perfect static/motion routing."
echo "  - Continue only if oracle_direction or oracle_block32 clearly exceeds current region/LTX baselines."
