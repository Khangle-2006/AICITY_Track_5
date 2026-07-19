#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/experiments/wan_ticft
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl}
OUT=${OUT:-$ROOT/seedance2_fast_ref_bdd_gate}
DEPTH_REPORT=${DEPTH_REPORT:-$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv}
LIMIT=${LIMIT:-2}
OFFSET=${OFFSET:-0}
MODE=${MODE:-seedance2_fast_ref}
RESOLUTION=${RESOLUTION:-480p}
DURATION=${DURATION:-4}
SEED=${SEED:-23}
export PYTHONPATH=/workspace:${PYTHONPATH:-}

echo "=== Seedance fal BDD gate ==="
echo "manifest=$MANIFEST"
echo "out=$OUT"
echo "limit=$LIMIT offset=$OFFSET mode=$MODE resolution=$RESOLUTION duration=$DURATION"

if [[ -z "${FAL_KEY:-}" ]]; then
  echo "FAL_KEY is not set. Run: export FAL_KEY='your-fal-key'" >&2
  exit 2
fi

python -m wan_ticft.evaluation.run_seedance_fal_bdd_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --limit "$LIMIT" \
  --offset "$OFFSET" \
  --mode "$MODE" \
  --resolution "$RESOLUTION" \
  --duration "$DURATION" \
  --seed "$SEED" \
  --clean

echo "[1/2] Pixel metrics"
python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports seedance="$OUT/report.csv" depth_a75="$DEPTH_REPORT" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "$OUT/eval_pixel.csv"

echo "[2/2] FID/FVD proxy"
python -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report seedance="$OUT/report.csv" \
  --report depth_a75="$DEPTH_REPORT" \
  --clip_frames 8 \
  --out_csv "$OUT/eval_proxy.csv"

echo
echo "report=$OUT/report.csv"
echo "pixel=$OUT/eval_pixel.csv"
echo "proxy=$OUT/eval_proxy.csv"
