#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
OUT="$ROOT/wts_distribution_gate20"
MANIFEST="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Materialize WTS distribution gate: 10 samples/domain, 50 future frames ==="
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --limit_per_domain 10 \
  --frames 50 \
  --clean

echo
echo "=== WTS relative FID/FVD proxy ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --target_report "$OUT/target_report.csv" \
  --full_video \
  --report "repeat_last=$OUT/repeat_last_report.csv" \
  --report "region_medium=$OUT/region_norm_medium_report.csv" \
  --report "region_long12=$OUT/region_norm_long12_report.csv" \
  --report "region_long24=$OUT/region_norm_long24_report.csv" \
  --out_csv "$ROOT/wts_distribution_proxy_gate20.csv"

echo
echo "Gate rule:"
echo "  - A WTS replacement must improve both FID and FVD proxies versus region_medium."
echo "  - It must already have acceptable PSNR/SSIM evidence from the existing"
echo "    long-horizon gate and pass visual QA."
echo "  - If none pass, retain region_medium and stop WTS flow-duration tuning."
