#!/usr/bin/env bash
set -euo pipefail

echo "=== Cosmos3 background-prior gate: 24 BDD samples ==="
echo "Step 1: generate/resume Cosmos3 strict image anchors."
echo "Step 2: inject unmasked low-frequency Cosmos background into cnn_ltxmix65."

cd /workspace
export PYTHONPATH=/workspace/wan_ticft:${PYTHONPATH:-}

SERVER=${SERVER:-http://127.0.0.1:8000}
MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
ANCHOR_OUT=${ANCHOR_OUT:-/workspace/experiments/wan_ticft/cosmos3_bdd_anchor_gate24}
HYBRID_OUT=${HYBRID_OUT:-/workspace/experiments/wan_ticft/cosmos3_background_hybrid_gate24}
BASE_REPORT=${BASE_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
BASE_LABEL=${BASE_LABEL:-cnn_ltxmix65}

STEPS=${STEPS:-28}
FRAMES=${FRAMES:-40}
FPS=${FPS:-30}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-720}
GUIDANCE=${GUIDANCE:-8.0}
FLOW_SHIFT=${FLOW_SHIFT:-8.0}

echo "server=$SERVER"
echo "manifest=$MANIFEST"
echo "anchor_out=$ANCHOR_OUT"
echo "hybrid_out=$HYBRID_OUT"
echo "base=$BASE_LABEL $BASE_REPORT"

python -m wan_ticft.evaluation.run_cosmos3_bdd_v2v_gate \
  --manifest "$MANIFEST" \
  --out_dir "$ANCHOR_OUT" \
  --server "$SERVER" \
  --limit 24 \
  --orientation landscape_cw \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --num_frames "$FRAMES" \
  --fps "$FPS" \
  --steps "$STEPS" \
  --guidance_scale "$GUIDANCE" \
  --flow_shift "$FLOW_SHIFT" \
  --conditioning_mode image \
  --resume

python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$ANCHOR_OUT/selected_manifest.jsonl" \
  --reports "cosmos3_anchor=$ANCHOR_OUT/report.csv" "transition16=/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv" "$BASE_LABEL=$BASE_REPORT" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${ANCHOR_OUT}_eval.csv"

python -m wan_ticft.evaluation.materialize_cosmos_background_hybrid \
  --base_report "$BASE_REPORT" \
  --cosmos_report "$ANCHOR_OUT/report.csv" \
  --out_root "$HYBRID_OUT" \
  --clean \
  --variant bg_s31_a25:31:0.25 \
  --variant bg_s31_a20:31:0.20 \
  --variant bg_s45_a25:45:0.25

python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$ANCHOR_OUT/selected_manifest.jsonl" \
  --reports \
    "$BASE_LABEL=$BASE_REPORT" \
    bg_s31_a25="$HYBRID_OUT/bg_s31_a25/report.csv" \
    bg_s31_a20="$HYBRID_OUT/bg_s31_a20/report.csv" \
    bg_s45_a25="$HYBRID_OUT/bg_s45_a25/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${HYBRID_OUT}_eval.csv"

for variant in bg_s31_a25 bg_s31_a20 bg_s45_a25; do
  python -m wan_ticft.evaluation.make_cosmos3_debug_sheets \
    --manifest "$ANCHOR_OUT/selected_manifest.jsonl" \
    --cosmos_report "$HYBRID_OUT/$variant/report.csv" \
    --transition_report "$BASE_REPORT" \
    --out_dir "${HYBRID_OUT}_${variant}_sheets"
done

cat <<EOF

Outputs:
  anchor videos: $ANCHOR_OUT/videos
  anchor eval: ${ANCHOR_OUT}_eval.csv
  hybrid videos: $HYBRID_OUT
  hybrid eval: ${HYBRID_OUT}_eval.csv
  sheets:
    ${HYBRID_OUT}_bg_s31_a25_sheets
    ${HYBRID_OUT}_bg_s31_a20_sheets
    ${HYBRID_OUT}_bg_s45_a25_sheets

Gate rule:
  - Promote only if bg_s31_a25 or a safer neighbor improves 24-sample PSNR/SSIM
    over cnn_ltxmix65 and visual sheets do not show severe actor/vehicle ghosting.
  - If 24-sample gain collapses or artifacts are broad, keep cnn_ltxmix65 and stop
    Cosmos background injection for submission.
EOF
