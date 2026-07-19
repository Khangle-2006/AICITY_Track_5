#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
OUT="$ROOT/ltx_region_lowfreq_s16_a50_val24"
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
TRANSITION_REPORT="$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Materialize best low-frequency candidate ==="
"$PYTHON" -m wan_ticft.evaluation.materialize_ltx_region_lowfreq \
  --manifest "$MANIFEST" \
  --ltx_report "$TRANSITION_REPORT" \
  --out_dir "$OUT" \
  --sigma 16 \
  --alpha 0.50 \
  --clean

echo
echo "=== Re-evaluate materialized video ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports "ltx_transition16=$TRANSITION_REPORT" "lowfreq_s16_a50=$OUT/report.csv" \
  --out_csv "$ROOT/ltx_region_lowfreq_s16_a50_materialized_eval.csv" \
  --offset_mode timestamp

echo
echo "=== Contact sheets ==="
"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/videos" \
  --mode videos \
  --out_dir "${OUT}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

echo
echo "=== Relative FID/FVD proxy audit ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report "ltx_raw=$ROOT/ltxv_bdd25_gate/report.csv" \
  --report "ltx_transition16=$TRANSITION_REPORT" \
  --report "lowfreq_s16_a50=$OUT/report.csv" \
  --report "wan=$ROOT/wan22_bdd25_seed101_s10/report.csv" \
  --out_csv "$ROOT/bdd_perceptual_video_distribution_proxy.csv"

echo
echo "Promotion rule:"
echo "  - Materialized lowfreq must retain >= +0.30 PSNR and non-negative SSIM"
echo "    versus transition16."
echo "  - It must improve both relative FID and FVD proxies versus transition16."
echo "  - Visual QA must show no new blur, ghosting, flicker, or motion suppression."
echo "  - Otherwise retain transition16 and close low-frequency correction."
