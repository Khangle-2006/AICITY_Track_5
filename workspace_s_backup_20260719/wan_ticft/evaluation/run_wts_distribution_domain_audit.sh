#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
OUT="$ROOT/wts_distribution_gate20"
MANIFEST="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

for VIEW in wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera; do
  SHORT="${VIEW#wts_}"
  echo "=== WTS relative FID/FVD proxy: $VIEW ==="
  /workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  TORCH_HOME="$ROOT/tools/torch_cache" \
  "$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
    --manifest "$MANIFEST" \
    --view_type "$VIEW" \
    --target_report "$OUT/target_report.csv" \
    --full_video \
    --report "repeat_last=$OUT/repeat_last_report.csv" \
    --report "region_medium=$OUT/region_norm_medium_report.csv" \
    --report "region_long12=$OUT/region_norm_long12_report.csv" \
    --report "region_long24=$OUT/region_norm_long24_report.csv" \
    --out_csv "$ROOT/wts_distribution_${SHORT}_proxy_gate10.csv"
done

echo
echo "Routing rule:"
echo "  - Route a domain to long12/long24 only if it improves both FID and FVD"
echo "    proxies versus region_medium in that domain."
echo "  - Otherwise retain region_medium for that domain."
echo "  - If neither domain passes, do not tune flow horizon further."
