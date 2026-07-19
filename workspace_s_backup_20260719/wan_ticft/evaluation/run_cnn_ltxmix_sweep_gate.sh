#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl}
LTX_REPORT=${LTX_REPORT:-$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv}
TRADEOFF_OUT=${TRADEOFF_OUT:-$ROOT/cnn_selector_perceptual_tradeoff_gate}
CNN0_REPORT=${CNN0_REPORT:-$TRADEOFF_OUT/cnn_ltxmix0/report.csv}
MIX65_REPORT=${MIX65_REPORT:-$TRADEOFF_OUT/cnn_ltxmix65/report.csv}
OUT=${OUT:-$ROOT/cnn_ltxmix_sweep_gate}
PIXEL_CSV=${PIXEL_CSV:-$ROOT/cnn_ltxmix_sweep_eval.csv}
PROXY_CSV=${PROXY_CSV:-$ROOT/cnn_ltxmix_sweep_proxy.csv}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== CNN/LTX mix sweep gate ==="
echo "Uses existing CNN validation report and transition16; no CNN retraining."

if [[ ! -f "$CNN0_REPORT" || ! -f "$MIX65_REPORT" ]]; then
  echo "Missing CNN validation reports; regenerating cnn_ltxmix0 and cnn_ltxmix65."
  "$PYTHON" -m wan_ticft.evaluation.materialize_cnn_ltx_region_selector_variants \
    --manifest "$MANIFEST" \
    --ltx_report "$LTX_REPORT" \
    --out_root "$TRADEOFF_OUT" \
    --epochs 8 \
    --batch_size 8 \
    --variant cnn_ltxmix0:ltx_mix:0.0 \
    --variant cnn_ltxmix65:ltx_mix:0.65 \
    --clean
fi

"$PYTHON" -m wan_ticft.evaluation.materialize_video_report_mix \
  --report_a "$CNN0_REPORT" \
  --report_b "$LTX_REPORT" \
  --out_root "$OUT" \
  --variant mix45=0.45 \
  --variant mix55=0.55 \
  --variant mix60=0.60 \
  --variant mix70=0.70 \
  --variant mix75=0.75 \
  --clean

echo
echo "=== Pixel metrics ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    transition16="$LTX_REPORT" \
    mix45="$OUT/mix45/report.csv" \
    mix55="$OUT/mix55/report.csv" \
    mix60="$OUT/mix60/report.csv" \
    mix65="$MIX65_REPORT" \
    mix70="$OUT/mix70/report.csv" \
    mix75="$OUT/mix75/report.csv" \
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
  --report "mix45=$OUT/mix45/report.csv" \
  --report "mix55=$OUT/mix55/report.csv" \
  --report "mix60=$OUT/mix60/report.csv" \
  --report "mix65=$MIX65_REPORT" \
  --report "mix70=$OUT/mix70/report.csv" \
  --report "mix75=$OUT/mix75/report.csv" \
  --out_csv "$PROXY_CSV"

echo
echo "Gate rule:"
echo "  - Choose the mix with the best expected leaderboard tradeoff:"
echo "    meaningful PSNR gain over transition16 and FID/FVD near mix65 or better."
echo "  - If mix60 has much higher pixel with tolerable proxy, build mix60."
echo "  - If mix70/75 improves proxy with modest pixel loss, build that instead."
