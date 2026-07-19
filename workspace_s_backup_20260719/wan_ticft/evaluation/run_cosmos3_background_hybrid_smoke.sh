#!/usr/bin/env bash
set -euo pipefail

echo "=== Cosmos3 low-frequency background hybrid smoke ==="
echo "Uses Cosmos3 anchor as background/color/low-frequency prior only."

cd /workspace
export PYTHONPATH=/workspace/wan_ticft:${PYTHONPATH:-}

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/cosmos3_bdd_anchor_correction4/selected_manifest.jsonl}
COSMOS_REPORT=${COSMOS_REPORT:-/workspace/experiments/wan_ticft/cosmos3_bdd_anchor_correction4/report.csv}
TRANSITION_REPORT=${TRANSITION_REPORT:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv}
CNN65_REPORT=${CNN65_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
OUT=${OUT:-/workspace/experiments/wan_ticft/cosmos3_background_hybrid_smoke4}

BASE_REPORT=$TRANSITION_REPORT
BASE_LABEL=transition16
if [[ -f "$CNN65_REPORT" ]]; then
  BASE_REPORT=$CNN65_REPORT
  BASE_LABEL=cnn_ltxmix65
fi

echo "manifest=$MANIFEST"
echo "base=$BASE_LABEL $BASE_REPORT"
echo "cosmos=$COSMOS_REPORT"
echo "out=$OUT"

python -m wan_ticft.evaluation.materialize_cosmos_background_hybrid \
  --base_report "$BASE_REPORT" \
  --cosmos_report "$COSMOS_REPORT" \
  --out_root "$OUT" \
  --clean \
  --variant bg_s31_a15:31:0.15 \
  --variant bg_s31_a25:31:0.25 \
  --variant bg_s45_a20:45:0.20

python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    "$BASE_LABEL=$BASE_REPORT" \
    cosmos3_anchor="$COSMOS_REPORT" \
    bg_s31_a15="$OUT/bg_s31_a15/report.csv" \
    bg_s31_a25="$OUT/bg_s31_a25/report.csv" \
    bg_s45_a20="$OUT/bg_s45_a20/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

for variant in bg_s31_a15 bg_s31_a25 bg_s45_a20; do
  python -m wan_ticft.evaluation.make_cosmos3_debug_sheets \
    --manifest "$MANIFEST" \
    --cosmos_report "$OUT/$variant/report.csv" \
    --transition_report "$BASE_REPORT" \
    --out_dir "${OUT}_${variant}_sheets"
done

cat <<EOF

Outputs:
  videos: $OUT
  eval: ${OUT}_eval.csv
  sheets:
    ${OUT}_bg_s31_a15_sheets
    ${OUT}_bg_s31_a25_sheets
    ${OUT}_bg_s45_a20_sheets

Gate rule:
  - Continue only if a background hybrid visibly keeps Cosmos background stability
    without importing hallucinated actors/objects.
  - It must beat or stay close to the base on PSNR/SSIM; a small pixel loss is
    acceptable only if visual background drift is clearly fixed.
  - If all variants inherit Cosmos hallucination or lower metrics hard, do not
    replace the base; use Cosmos only as a diagnostic direction.
EOF
