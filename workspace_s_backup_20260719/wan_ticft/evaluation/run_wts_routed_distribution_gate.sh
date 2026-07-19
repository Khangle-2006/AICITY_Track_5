#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
OUT="$ROOT/wts_distribution_gate20"
MANIFEST="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
ROUTED="$OUT/region_overhead_long24_vehicle_medium_report.csv"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

"$PYTHON" -m wan_ticft.evaluation.build_wts_routed_distribution_report \
  --medium_report "$OUT/region_norm_medium_report.csv" \
  --long24_report "$OUT/region_norm_long24_report.csv" \
  --out_report "$ROUTED"

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --target_report "$OUT/target_report.csv" \
  --full_video \
  --report "region_medium=$OUT/region_norm_medium_report.csv" \
  --report "wts_routed=$ROUTED" \
  --out_csv "$ROOT/wts_routed_distribution_proxy_gate20.csv"

echo
echo "Final routing gate:"
echo "  - Pass only if wts_routed improves both aggregate FID and FVD proxies."
echo "  - Overhead long24 already improves long-horizon PSNR and SSIM."
echo "  - If pass, perform visual QA before building the test submission."
