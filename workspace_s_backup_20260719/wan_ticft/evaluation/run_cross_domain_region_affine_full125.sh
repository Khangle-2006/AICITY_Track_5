#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft

echo "=== Cross-domain history camera-motion blend: full balanced validation ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_manifest_motion_baselines \
  --manifest "$ROOT/manifests/val_ticft_balanced_125_frames.jsonl" \
  --out_csv "$ROOT/cross_domain_region_affine_full125.csv" \
  --candidates region_norm_medium region_affine_g20_history

echo
echo "Promotion rule:"
echo "  - Promote only if the camera blend improves macro PSNR,"
echo "    does not reduce macro SSIM, and no domain loses >0.05 PSNR."
echo "  - If pass, run visual QA before building any test candidate."
echo "  - If fail, retain region_norm_medium. No weight sweep."
