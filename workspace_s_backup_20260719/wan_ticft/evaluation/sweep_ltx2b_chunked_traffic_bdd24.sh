#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=${ROOT:-/workspace/experiments/wan_ticft}
SIZES="${SIZES:-8 12 16 24}"
CUDA_ID="${CUDA_VISIBLE_DEVICES:-2}"

echo "=== Sweep LTX2B chunked traffic BDD24 ==="
echo "sizes=$SIZES"
echo "cuda=$CUDA_ID"

for size in $SIZES; do
  safe_size="${size//./p}"
  out="$ROOT/ltx2b_chunked_traffic_bdd24_c${safe_size}"
  echo
  echo ">>> CHUNK_FRAMES=$size OUT=$out"
  OUT="$out" CHUNK_FRAMES="$size" CUDA_VISIBLE_DEVICES="$CUDA_ID" \
    bash /workspace/wan_ticft/evaluation/run_ltx2b_chunked_traffic_bdd24_gate.sh
done

echo
echo "Sweep complete. CSVs:"
for size in $SIZES; do
  safe_size="${size//./p}"
  out="$ROOT/ltx2b_chunked_traffic_bdd24_c${safe_size}"
  echo "  c$size pixel=$out/eval_pixel.csv proxy=$out/eval_proxy.csv"
done
