#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl}
LTX_REPORT=${LTX_REPORT:-$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv}
OUT=${OUT:-$ROOT/cnn_selector_perceptual_tradeoff_gate}
PIXEL_CSV=${PIXEL_CSV:-$ROOT/cnn_selector_perceptual_tradeoff_eval.csv}
PROXY_CSV=${PROXY_CSV:-$ROOT/cnn_selector_perceptual_tradeoff_proxy.csv}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== CNN selector perceptual tradeoff gate ==="
echo "Goal: keep CNN pixel gain but move distribution back toward LTX to reduce FID/FVD."

"$PYTHON" -m wan_ticft.evaluation.materialize_cnn_ltx_region_selector_variants \
  --manifest "$MANIFEST" \
  --ltx_report "$LTX_REPORT" \
  --out_root "$OUT" \
  --epochs 8 \
  --batch_size 8 \
  --variant cnn_ltxmix0:ltx_mix:0.0 \
  --variant cnn_ltxmix25:ltx_mix:0.25 \
  --variant cnn_ltxmix50:ltx_mix:0.50 \
  --variant cnn_ltxmix65:ltx_mix:0.65 \
  --clean

echo
echo "=== Pixel metrics ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    transition16="$LTX_REPORT" \
    cnn_ltxmix0="$OUT/cnn_ltxmix0/report.csv" \
    cnn_ltxmix25="$OUT/cnn_ltxmix25/report.csv" \
    cnn_ltxmix50="$OUT/cnn_ltxmix50/report.csv" \
    cnn_ltxmix65="$OUT/cnn_ltxmix65/report.csv" \
  --offset_mode timestamp \
  --out_csv "$PIXEL_CSV"

echo
echo "=== Relative FID/FVD proxy ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report "transition16=$LTX_REPORT" \
  --report "cnn_ltxmix0=$OUT/cnn_ltxmix0/report.csv" \
  --report "cnn_ltxmix25=$OUT/cnn_ltxmix25/report.csv" \
  --report "cnn_ltxmix50=$OUT/cnn_ltxmix50/report.csv" \
  --report "cnn_ltxmix65=$OUT/cnn_ltxmix65/report.csv" \
  --out_csv "$PROXY_CSV"

echo
echo "Gate rule:"
echo "  - Promote only if a mixed CNN candidate keeps >= +0.30 PSNR over transition16,"
echo "    and improves both FID and FVD proxy versus raw CNN."
echo "  - Prefer the strongest LTX mix that still keeps a meaningful PSNR/SSIM gain."
echo "  - If all still have bad proxy, CNN remains pixel-only and should not be the main candidate."
