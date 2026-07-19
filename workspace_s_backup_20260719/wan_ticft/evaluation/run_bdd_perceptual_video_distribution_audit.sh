#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== BDD perceptual/video distribution proxy audit ==="
echo "This ranks candidates locally; values are not official leaderboard FID/FVD."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl" \
  --report "ltx_raw=$ROOT/ltxv_bdd25_gate/report.csv" \
  --report "ltx_transition16=$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv" \
  --report "wan=$ROOT/wan22_bdd25_seed101_s10/report.csv" \
  --out_csv "$ROOT/bdd_perceptual_video_distribution_proxy.csv"

echo
echo "Audit rule:"
echo "  - Trust this proxy only if transition16 ranks better than raw LTX, matching"
echo "    the known leaderboard improvement."
echo "  - Any future FID/FVD-targeted correction must improve both proxies without"
echo "    a material PSNR/SSIM regression before submission."
