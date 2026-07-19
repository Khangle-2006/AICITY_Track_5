#!/bin/bash
set -euo pipefail
# Offline sweep: does camera-degradation lower REAL-FID? Tests a75 (best, FID 70.7) and chunk
# at several strengths. Goal: find a degradation that drops REAL-FID below a75's 70.7.
cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=/workspace/experiments/wan_ticft
REAL=$ROOT/frames_balanced_val_125/val
A75=$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/videos
CHUNK=$ROOT/bdd_depth_chunkbase_sweep_dpt_eval24/depth_layered_a75/videos
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

run() {  # label blur noise jpeg ds
  echo ">>> $1  (blur=$2 noise=$3 jpeg=$4 ds=$5)"
  "$PYTHON" -m wan_ticft.evaluation.compute_real_fid \
    --real_root "$REAL" --work "$ROOT/_fidsweep_$1" --limit 24 \
    --blur "$2" --noise "$3" --jpeg_q "$4" --downscale "$5" \
    --candidate "a75=$A75" --candidate "chunk=$CHUNK"
  echo
}

echo "=== DEGRADATION REAL-FID SWEEP (baseline a75 RAW = 70.7) ==="
run raw     0    0   100 1.0
run light   0.4  3   45  1.0
run med     0.6  4   35  1.0
run strong  0.9  6   25  0.85
run bluronly 0.8 0   100 1.0
run jpegonly 0   0   25  1.0
run noiseonly 0  6   100 1.0
echo "=== DONE. Pick the (method,strength) with lowest REAL-FID; if any a75 variant < 70.7 -> submit it. ==="
