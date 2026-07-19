#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft

echo "=== Cross-domain future-warp and hole-headroom gate: 10 samples/domain ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_cross_domain_warp_hole_oracle \
  --manifest "$ROOT/manifests/val_ticft_balanced_125_frames.jsonl" \
  --out_csv "$ROOT/cross_domain_warp_hole_oracle_gate30.csv" \
  --flow_width 640 \
  --hole_dilate 0 4 \
  --limit_per_domain 10

echo
echo "Final oracle stop rule:"
echo "  - Pass only if target-hole d4 beats same-mask repeat fill by >=0.50 PSNR"
echo "    in every domain and d4 mask coverage stays below 25%."
echo "  - If pass: stop oracle work and test one deployable future-flow approach."
echo "  - If fail: close the general warp-and-inpaint direction."
