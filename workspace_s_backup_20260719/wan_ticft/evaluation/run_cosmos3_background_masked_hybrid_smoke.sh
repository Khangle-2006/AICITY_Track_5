#!/usr/bin/env bash
set -euo pipefail

echo "=== Cosmos3 masked background hybrid smoke ==="
echo "Only injects Cosmos3 low-frequency prior in low-motion / low-edge background regions."

cd /workspace
export PYTHONPATH=/workspace/wan_ticft:${PYTHONPATH:-}

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/cosmos3_bdd_anchor_correction4/selected_manifest.jsonl}
COSMOS_REPORT=${COSMOS_REPORT:-/workspace/experiments/wan_ticft/cosmos3_bdd_anchor_correction4/report.csv}
CNN65_REPORT=${CNN65_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
BASE_REPORT=${BASE_REPORT:-$CNN65_REPORT}
BASE_LABEL=${BASE_LABEL:-cnn_ltxmix65}
UNMASKED_REPORT=${UNMASKED_REPORT:-/workspace/experiments/wan_ticft/cosmos3_background_hybrid_smoke4/bg_s31_a25/report.csv}
OUT=${OUT:-/workspace/experiments/wan_ticft/cosmos3_background_masked_hybrid_smoke4}

echo "manifest=$MANIFEST"
echo "base=$BASE_LABEL $BASE_REPORT"
echo "cosmos=$COSMOS_REPORT"
echo "out=$OUT"

python -m wan_ticft.evaluation.materialize_cosmos_background_masked_hybrid \
  --base_report "$BASE_REPORT" \
  --cosmos_report "$COSMOS_REPORT" \
  --out_root "$OUT" \
  --clean \
  --variant bgmask_s31_a25_m03_e40:31:0.25:0.03:40 \
  --variant bgmask_s31_a30_m04_e50:31:0.30:0.04:50 \
  --variant bgmask_s45_a25_m03_e40:45:0.25:0.03:40

REPORTS=(
  "$BASE_LABEL=$BASE_REPORT"
  "cosmos_bg_unmasked=$UNMASKED_REPORT"
  "bgmask_s31_a25_m03_e40=$OUT/bgmask_s31_a25_m03_e40/report.csv"
  "bgmask_s31_a30_m04_e50=$OUT/bgmask_s31_a30_m04_e50/report.csv"
  "bgmask_s45_a25_m03_e40=$OUT/bgmask_s45_a25_m03_e40/report.csv"
)

python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports "${REPORTS[@]}" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

for variant in bgmask_s31_a25_m03_e40 bgmask_s31_a30_m04_e50 bgmask_s45_a25_m03_e40; do
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
    ${OUT}_bgmask_s31_a25_m03_e40_sheets
    ${OUT}_bgmask_s31_a30_m04_e50_sheets
    ${OUT}_bgmask_s45_a25_m03_e40_sheets

Gate rule:
  - Promote a masked variant only if it keeps most of unmasked bg_s31_a25's
    metric gain and visibly reduces foreground/actor ghosting on 3237.
  - If mask removes the metric gain, keep unmasked for one 24-sample gate but
    inspect foreground artifacts before any submission.
EOF
