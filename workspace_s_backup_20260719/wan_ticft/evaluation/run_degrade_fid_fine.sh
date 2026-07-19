#!/bin/bash
set -euo pipefail
# Fine sweep around the winning "light" point, a75 only, to minimize REAL-FID (baseline raw=70.7, light=69.1).
cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=/workspace/experiments/wan_ticft
REAL=$ROOT/frames_balanced_val_125/val
A75=$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/videos
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
run() {
  echo ">>> $1 (blur=$2 noise=$3 jpeg=$4)"
  "$PYTHON" -m wan_ticft.evaluation.compute_real_fid --real_root "$REAL" --work "$ROOT/_fidfine_$1" \
    --limit 24 --blur "$2" --noise "$3" --jpeg_q "$4" --downscale 1.0 --candidate "a75=$A75"
  echo
}
echo "=== FINE SWEEP around light (a75; raw=70.7, light=69.1) ==="
run vlight   0.3 2 50
run lightA   0.4 3 45
run lightB   0.4 2 40
run lightC   0.5 3 40
run lightD   0.3 3 45
run lightE   0.5 2 45
run lightF   0.4 3 50
echo "=== Pick lowest; that's the a75 degradation to build on test + submit. ==="
