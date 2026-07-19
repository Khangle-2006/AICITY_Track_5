#!/usr/bin/env bash
set -euo pipefail

echo "=== Cosmos3 official-style 5-frame BDD V2V smoke ==="
echo "Uses the last 5 history frames as video input, matching Cosmos3's documented video-input limit."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
export PYTHONPATH=/workspace/wan_ticft:${PYTHONPATH:-}

SERVER=${SERVER:-http://127.0.0.1:8000}
MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
OUT=${OUT:-/workspace/experiments/wan_ticft/cosmos3_bdd_5frame_v2v_smoke4}
HYBRID_OUT=${HYBRID_OUT:-/workspace/experiments/wan_ticft/cosmos3_bdd_5frame_bg_hybrid_smoke4}
BASE_REPORT=${BASE_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
TRANSITION_REPORT=${TRANSITION_REPORT:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv}
LIMIT=${LIMIT:-4}

STEPS=${STEPS:-28}
FRAMES=${FRAMES:-40}
FPS=${FPS:-30}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-720}
GUIDANCE=${GUIDANCE:-8.0}
FLOW_SHIFT=${FLOW_SHIFT:-8.0}

echo "server=$SERVER"
echo "manifest=$MANIFEST"
echo "out=$OUT"
echo "hybrid_out=$HYBRID_OUT"
echo "limit=$LIMIT"

"$PYTHON" -m wan_ticft.evaluation.run_cosmos3_bdd_v2v_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --server "$SERVER" \
  --limit "$LIMIT" \
  --history_frames 5 \
  --orientation landscape_cw \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --num_frames "$FRAMES" \
  --fps "$FPS" \
  --steps "$STEPS" \
  --guidance_scale "$GUIDANCE" \
  --flow_shift "$FLOW_SHIFT" \
  --conditioning_mode video \
  --condition_frame_indexes 0 1 2 3 4 \
  --condition_video_keep last \
  --resume

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cosmos5_v2v="$OUT/report.csv" \
    cnn_ltxmix65="$BASE_REPORT" \
    transition16="$TRANSITION_REPORT" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.materialize_cosmos_background_hybrid \
  --base_report "$BASE_REPORT" \
  --cosmos_report "$OUT/report.csv" \
  --out_root "$HYBRID_OUT" \
  --clean \
  --variant bg5_s31_a10:31:0.10 \
  --variant bg5_s31_a15:31:0.15 \
  --variant bg5_s31_a25:31:0.25

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    bg5_s31_a10="$HYBRID_OUT/bg5_s31_a10/report.csv" \
    bg5_s31_a15="$HYBRID_OUT/bg5_s31_a15/report.csv" \
    bg5_s31_a25="$HYBRID_OUT/bg5_s31_a25/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${HYBRID_OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report bg5_s31_a10="$HYBRID_OUT/bg5_s31_a10/report.csv" \
  --report bg5_s31_a15="$HYBRID_OUT/bg5_s31_a15/report.csv" \
  --report bg5_s31_a25="$HYBRID_OUT/bg5_s31_a25/report.csv" \
  --out_csv "${HYBRID_OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

for report in "$OUT/report.csv" "$HYBRID_OUT/bg5_s31_a15/report.csv" "$HYBRID_OUT/bg5_s31_a25/report.csv"; do
  label=$(basename "$(dirname "$report")")
  "$PYTHON" -m wan_ticft.evaluation.make_cosmos3_debug_sheets \
    --manifest "$OUT/selected_manifest.jsonl" \
    --cosmos_report "$report" \
    --transition_report "$BASE_REPORT" \
    --out_dir "${HYBRID_OUT}_${label}_sheets"
done

cat <<EOF

Outputs:
  raw videos: $OUT/videos
  raw eval: ${OUT}_eval.csv
  hybrid videos: $HYBRID_OUT
  hybrid eval: ${HYBRID_OUT}_eval.csv
  hybrid proxy: ${HYBRID_OUT}_proxy.csv
  sheets:
    ${HYBRID_OUT}_$(basename "$OUT")_sheets
    ${HYBRID_OUT}_bg5_s31_a15_sheets
    ${HYBRID_OUT}_bg5_s31_a25_sheets

Gate rule:
  - Continue only if 5-frame V2V reduces background ghost/drift versus image-anchor
    and bg5 proxy is not worse than the image-anchor alpha sweep.
  - If raw Cosmos5 changes scene or bg5 still worsens FID/FVD, close Cosmos background
    for leaderboard and keep cnn_ltxmix65.
EOF
