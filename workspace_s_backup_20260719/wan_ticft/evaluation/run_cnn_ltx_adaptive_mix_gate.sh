#!/usr/bin/env bash
set -euo pipefail

echo "=== CNN/LTX adaptive mix gate ==="
echo "Goal: reduce FID/FVD without scene changes, using only CNN and LTX."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
export PYTHONPATH=/workspace/wan_ticft:${PYTHONPATH:-}

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
CNN_REPORT=${CNN_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix0/report.csv}
LTX_REPORT=${LTX_REPORT:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv}
MIX65_REPORT=${MIX65_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
OUT=${OUT:-/workspace/experiments/wan_ticft/cnn_ltx_adaptive_mix_gate24}

echo "manifest=$MANIFEST"
echo "cnn=$CNN_REPORT"
echo "ltx=$LTX_REPORT"
echo "mix65=$MIX65_REPORT"
echo "out=$OUT"

"$PYTHON" -m wan_ticft.evaluation.materialize_cnn_ltx_adaptive_mix \
  --cnn_report "$CNN_REPORT" \
  --ltx_report "$LTX_REPORT" \
  --out_root "$OUT" \
  --clean \
  --variant ramp45_70_t24:ramp:0.45:0.70:24 \
  --variant ramp55_75_t24:ramp:0.55:0.75:24 \
  --variant motion45_75_t24:motion:0.45:0.75:24 \
  --variant edge45_75_t24:edge:0.45:0.75:24 \
  --variant motion55_80_t32:motion:0.55:0.80:32

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    cnn_ltxmix65="$MIX65_REPORT" \
    transition16="$LTX_REPORT" \
    ramp45_70_t24="$OUT/ramp45_70_t24/report.csv" \
    ramp55_75_t24="$OUT/ramp55_75_t24/report.csv" \
    motion45_75_t24="$OUT/motion45_75_t24/report.csv" \
    edge45_75_t24="$OUT/edge45_75_t24/report.csv" \
    motion55_80_t32="$OUT/motion55_80_t32/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report cnn_ltxmix65="$MIX65_REPORT" \
  --report transition16="$LTX_REPORT" \
  --report ramp45_70_t24="$OUT/ramp45_70_t24/report.csv" \
  --report ramp55_75_t24="$OUT/ramp55_75_t24/report.csv" \
  --report motion45_75_t24="$OUT/motion45_75_t24/report.csv" \
  --report edge45_75_t24="$OUT/edge45_75_t24/report.csv" \
  --report motion55_80_t32="$OUT/motion55_80_t32/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

for variant in ramp45_70_t24 ramp55_75_t24 motion45_75_t24 edge45_75_t24 motion55_80_t32; do
  "$PYTHON" -m baseline.video_contact_sheet \
    --input_dir "$OUT/$variant/videos" \
    --out_dir "${OUT}_${variant}_sheets" \
    --mode videos \
    --samples_per_input 16 || true
done

cat <<EOF

Outputs:
  pixel: ${OUT}_eval.csv
  proxy: ${OUT}_proxy.csv
  sheets: ${OUT}_*_sheets

Gate rule:
  - Continue only if a variant improves FID/FVD proxy versus cnn_ltxmix65
    while keeping meaningful PSNR/SSIM gain over transition16.
  - If all variants worsen proxy or lose too much pixel, keep cnn_ltxmix65.
EOF
