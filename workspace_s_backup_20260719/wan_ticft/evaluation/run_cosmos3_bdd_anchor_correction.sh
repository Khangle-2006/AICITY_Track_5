#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
OUT=${OUT:-/workspace/experiments/wan_ticft/cosmos3_bdd_anchor_correction4}
MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
SERVER=${SERVER:-http://127.0.0.1:8000}
LIMIT=${LIMIT:-4}
ORIENTATION=${ORIENTATION:-landscape_cw}
STEPS=${STEPS:-28}
FRAMES=${FRAMES:-40}
FPS=${FPS:-30}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-720}
GUIDANCE=${GUIDANCE:-8.0}
FLOW_SHIFT=${FLOW_SHIFT:-8.0}

echo "=== Cosmos3 BDD strict image-anchor correction ==="
echo "server=$SERVER"
echo "out=$OUT"
echo "orientation=$ORIENTATION"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python -m wan_ticft.evaluation.run_cosmos3_bdd_v2v_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --server "$SERVER" \
  --limit "$LIMIT" \
  --orientation "$ORIENTATION" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --num_frames "$FRAMES" \
  --fps "$FPS" \
  --steps "$STEPS" \
  --guidance_scale "$GUIDANCE" \
  --flow_shift "$FLOW_SHIFT" \
  --conditioning_mode image

echo
echo "=== Pixel eval versus transition16/repeat-last ==="
python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports "cosmos3_anchor=$OUT/report.csv" "transition16=/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

python -m wan_ticft.evaluation.make_cosmos3_debug_sheets \
  --manifest "$OUT/selected_manifest.jsonl" \
  --cosmos_report "$OUT/report.csv" \
  --out_dir "${OUT}_debug_sheets"

echo
echo "Outputs:"
echo "  videos: $OUT/videos"
echo "  eval: ${OUT}_eval.csv"
echo "  sheets: ${OUT}_debug_sheets"
echo
echo "Correction rule:"
echo "  - Continue Cosmos3 only if anchor mode keeps the same scene/object identities"
echo "    on the sheets. Pixel metrics may lag, but scene replacement is a hard fail."
