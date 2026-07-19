#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=${ROOT:-/workspace/experiments/wan_ticft}
OUT=${OUT:-$ROOT/ltx2b_dev40_chunked_traffic_bdd4_gate}
PIPELINE_CONFIG=${PIPELINE_CONFIG:-/workspace/third_party/LTX-Video/configs/ltxv-2b-0.9.6-dev.yaml}
LIMIT=${LIMIT:-4}
CHUNK_FRAMES=${CHUNK_FRAMES:-24}

echo "=== LTX2B dev40 chunked BDD4 gate ==="
echo "out=$OUT"
echo "config=$PIPELINE_CONFIG"
echo "limit=$LIMIT chunk_frames=$CHUNK_FRAMES"

OUT="$OUT" \
PIPELINE_CONFIG="$PIPELINE_CONFIG" \
LIMIT="$LIMIT" \
CHUNK_FRAMES="$CHUNK_FRAMES" \
bash /workspace/wan_ticft/evaluation/run_ltx2b_chunked_traffic_bdd4_gate.sh

echo "[extra] Full-horizon audit"
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports_full_horizon \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    ltx2b_dev40_chunked="$OUT/report.csv" \
    depth_a75="$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv" \
  --include_repeat_last \
  --out_csv "$OUT/eval_full_horizon.csv"

echo
echo "report=$OUT/report.csv"
echo "sparse=$OUT/eval_pixel.csv"
echo "full_horizon=$OUT/eval_full_horizon.csv"
echo "proxy=$OUT/eval_proxy.csv"
