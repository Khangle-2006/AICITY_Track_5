#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft

echo "=== Cross-domain history camera-motion blend smoke: 2 samples/domain ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_manifest_motion_baselines \
  --manifest "$ROOT/manifests/val_ticft_balanced_125_frames.jsonl" \
  --out_csv "$ROOT/cross_domain_region_affine_smoke6.csv" \
  --candidates region_norm_medium region_affine_g20_history \
  --limit_per_domain 2

echo
echo "Gate rule:"
echo "  - Pass only if the camera blend improves macro PSNR by >=0.10,"
echo "    does not reduce macro SSIM, and no domain loses >0.05 PSNR."
echo "  - If fail, close history-derived global camera blending. No weight sweep."
