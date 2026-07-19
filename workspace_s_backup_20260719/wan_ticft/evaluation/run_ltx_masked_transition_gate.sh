#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl}
RAW_LTX=${RAW_LTX:-$ROOT/ltxv_bdd25_gate/report.csv}
TRANSITION16=${TRANSITION16:-$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv}
OUT=${OUT:-$ROOT/ltx_masked_transition_gate}
PIXEL_CSV=${PIXEL_CSV:-$ROOT/ltx_masked_transition_gate_eval.csv}
PROXY_CSV=${PROXY_CSV:-$ROOT/ltx_masked_transition_gate_perceptual_proxy.csv}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== BDD LTX motion/edge masked transition gate ==="
echo "Goal: reduce early transition ghosting while preserving transition16 metrics."

"$PYTHON" -m wan_ticft.evaluation.materialize_ltx_masked_transition \
  --manifest "$MANIFEST" \
  --ltx_report "$RAW_LTX" \
  --out_root "$OUT" \
  --transition 16 \
  --variant mask_a25=0.25 \
  --variant mask_a40=0.40 \
  --variant mask_a60=0.60 \
  --clean

echo
echo "=== Pixel metrics ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    transition16="$TRANSITION16" \
    mask_a25="$OUT/mask_a25/report.csv" \
    mask_a40="$OUT/mask_a40/report.csv" \
    mask_a60="$OUT/mask_a60/report.csv" \
  --offset_mode timestamp \
  --out_csv "$PIXEL_CSV"

echo
echo "=== Relative FID/FVD proxy ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report "transition16=$TRANSITION16" \
  --report "mask_a25=$OUT/mask_a25/report.csv" \
  --report "mask_a40=$OUT/mask_a40/report.csv" \
  --report "mask_a60=$OUT/mask_a60/report.csv" \
  --out_csv "$PROXY_CSV"

echo
echo "=== Contact sheets for visual ghosting QA ==="
for name in mask_a25 mask_a40 mask_a60; do
  "$PYTHON" -m baseline.video_contact_sheet \
    --input "$OUT/$name/videos" \
    --mode videos \
    --out_dir "$OUT/${name}_sheets" \
    --samples 16 \
    --tile_width 320 \
    --cols 4
done

echo
echo "Gate rule:"
echo "  - Continue only if a mask variant visibly reduces early double exposure."
echo "  - It must not lose more than 0.10 PSNR / 0.003 SSIM versus transition16."
echo "  - It must improve FVD proxy or keep FVD within +0.15 while improving visual QA."
echo "  - If all fail, retain transition16 and close masked transition."
