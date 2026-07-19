#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=${ROOT:-/workspace/experiments/wan_ticft}
PROMPT_MODES="${PROMPT_MODES:-traffic_detail geometry_lock copy_bias dynamic_geometry}"
CHUNK_FRAMES="${CHUNK_FRAMES:-16}"
CUDA_ID="${CUDA_VISIBLE_DEVICES:-2}"

echo "=== Sweep LTX2B chunked prompt modes on BDD24 ==="
echo "prompt_modes=$PROMPT_MODES"
echo "chunk_frames=$CHUNK_FRAMES cuda=$CUDA_ID"

for mode in $PROMPT_MODES; do
  out="$ROOT/ltx2b_chunked_bdd24_prompt_${mode}_c${CHUNK_FRAMES}"
  echo
  echo ">>> PROMPT_MODE=$mode OUT=$out"
  OUT="$out" PROMPT_MODE="$mode" CHUNK_FRAMES="$CHUNK_FRAMES" CUDA_VISIBLE_DEVICES="$CUDA_ID" \
    bash /workspace/wan_ticft/evaluation/run_ltx2b_chunked_traffic_bdd24_gate.sh
done

echo
echo "Sweep complete. CSVs:"
for mode in $PROMPT_MODES; do
  out="$ROOT/ltx2b_chunked_bdd24_prompt_${mode}_c${CHUNK_FRAMES}"
  echo "  $mode pixel=$out/eval_pixel.csv proxy=$out/eval_proxy.csv"
done
