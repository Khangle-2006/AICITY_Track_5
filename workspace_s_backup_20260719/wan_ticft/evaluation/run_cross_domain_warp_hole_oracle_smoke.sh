#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft

echo "=== Cross-domain future-warp and hole-headroom smoke ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_cross_domain_warp_hole_oracle \
  --manifest "$ROOT/manifests/val_ticft_balanced_125_frames.jsonl" \
  --out_csv "$ROOT/cross_domain_warp_hole_oracle_smoke6.csv" \
  --flow_width 640 \
  --hole_dilate 0 4 8 \
  --limit_per_domain 2

echo
echo "Stop rule:"
echo "  - Stop the general warp-and-inpaint direction if target-hole d4 improves"
echo "    same-mask repeat fill by <0.50 PSNR in any domain."
echo "  - Continue to a larger balanced gate only if all three domains pass and"
echo "    useful d4 masks remain below 25%."
