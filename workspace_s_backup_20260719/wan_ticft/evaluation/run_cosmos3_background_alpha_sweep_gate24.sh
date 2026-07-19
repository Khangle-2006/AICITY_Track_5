#!/usr/bin/env bash
set -euo pipefail

echo "=== Cosmos3 background alpha sweep on 24-sample BDD gate ==="
echo "Purpose: reduce FID/FVD/CLIP damage from alpha=0.25 while keeping some pixel gain."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
export PYTHONPATH=/workspace/wan_ticft:${PYTHONPATH:-}

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/cosmos3_bdd_anchor_gate24/selected_manifest.jsonl}
COSMOS_REPORT=${COSMOS_REPORT:-/workspace/experiments/wan_ticft/cosmos3_bdd_anchor_gate24/report.csv}
BASE_REPORT=${BASE_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
OUT=${OUT:-/workspace/experiments/wan_ticft/cosmos3_background_alpha_sweep_gate24}

echo "manifest=$MANIFEST"
echo "cosmos=$COSMOS_REPORT"
echo "base=$BASE_REPORT"
echo "out=$OUT"

"$PYTHON" -m wan_ticft.evaluation.materialize_cosmos_background_hybrid \
  --base_report "$BASE_REPORT" \
  --cosmos_report "$COSMOS_REPORT" \
  --out_root "$OUT" \
  --clean \
  --variant bg_s31_a10:31:0.10 \
  --variant bg_s31_a15:31:0.15 \
  --variant bg_s31_a20:31:0.20 \
  --variant bg_s31_a25:31:0.25

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    bg_s31_a10="$OUT/bg_s31_a10/report.csv" \
    bg_s31_a15="$OUT/bg_s31_a15/report.csv" \
    bg_s31_a20="$OUT/bg_s31_a20/report.csv" \
    bg_s31_a25="$OUT/bg_s31_a25/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report bg_s31_a10="$OUT/bg_s31_a10/report.csv" \
  --report bg_s31_a15="$OUT/bg_s31_a15/report.csv" \
  --report bg_s31_a20="$OUT/bg_s31_a20/report.csv" \
  --report bg_s31_a25="$OUT/bg_s31_a25/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

cat <<EOF

Outputs:
  pixel: ${OUT}_eval.csv
  proxy: ${OUT}_proxy.csv

Decision:
  - If a10/a15 keeps proxy near cnn_ltxmix65 while still improving pixel, build that for submit.
  - If all alphas worsen proxy, stop Cosmos background for leaderboard and keep cnn_ltxmix65.
EOF
