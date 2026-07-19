#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
OUT=${OUT:-/workspace/experiments/wan_ticft/cosmos3_bdd_v2v_smoke4}
MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
SERVER=${SERVER:-http://127.0.0.1:8000}
LIMIT=${LIMIT:-4}
ORIENTATION=${ORIENTATION:-landscape_cw}
STEPS=${STEPS:-20}
FRAMES=${FRAMES:-49}
FPS=${FPS:-24}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-720}

echo "=== Cosmos3 BDD V2V smoke ==="
echo "server=$SERVER"
echo "out=$OUT"
echo "orientation=$ORIENTATION"
echo "Requires an already-running Cosmos3 vLLM-Omni server on port 8000."

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
  --condition_video_keep last \
  --condition_frame_indexes 0 1

echo
echo "=== Pixel eval versus transition16/repeat-last ==="
python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports "cosmos3=$OUT/report.csv" "transition16=/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo
echo "Outputs:"
echo "  videos: $OUT/videos"
echo "  report: $OUT/report.csv"
echo "  eval: ${OUT}_eval.csv"
echo
echo "Gate rule:"
echo "  - Continue only if visual QA is clearly better than LTX transition16"
echo "    and Cosmos3 does not catastrophically lose pixel metrics."
echo "  - If orientation is wrong, rerun with ORIENTATION=landscape_ccw before judging."
