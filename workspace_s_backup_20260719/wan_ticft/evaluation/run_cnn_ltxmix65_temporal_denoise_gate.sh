#!/usr/bin/env bash
set -euo pipefail

echo "=== CNN/LTX mix65 temporal denoise gate ==="
echo "No geometry warp. Reject unless FID/FVD proxy improves without large pixel loss."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
export PYTHONPATH=/workspace/wan_ticft:${PYTHONPATH:-}

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
BASE_REPORT=${BASE_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
OUT=${OUT:-/workspace/experiments/wan_ticft/cnn_ltxmix65_temporal_denoise_gate24}

"$PYTHON" -m wan_ticft.evaluation.materialize_temporal_denoise_report \
  --report "$BASE_REPORT" \
  --out_root "$OUT" \
  --clean \
  --variant td_a08_t04:0.08:0.04 \
  --variant td_a12_t04:0.12:0.04 \
  --variant td_a12_t06:0.12:0.06 \
  --variant td_a18_t06:0.18:0.06

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    td_a08_t04="$OUT/td_a08_t04/report.csv" \
    td_a12_t04="$OUT/td_a12_t04/report.csv" \
    td_a12_t06="$OUT/td_a12_t06/report.csv" \
    td_a18_t06="$OUT/td_a18_t06/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report td_a08_t04="$OUT/td_a08_t04/report.csv" \
  --report td_a12_t04="$OUT/td_a12_t04/report.csv" \
  --report td_a12_t06="$OUT/td_a12_t06/report.csv" \
  --report td_a18_t06="$OUT/td_a18_t06/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

cat <<EOF

Outputs:
  pixel: ${OUT}_eval.csv
  proxy: ${OUT}_proxy.csv

Gate rule:
  - Continue only if a denoise variant improves FID and FVD proxy versus cnn_ltxmix65
    and loses <0.05 PSNR / <0.002 SSIM.
  - If proxy worsens, close temporal denoise.
EOF
