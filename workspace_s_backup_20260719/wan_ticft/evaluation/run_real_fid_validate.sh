#!/bin/bash
set -euo pipefail
# Validate a trustworthy local FID against the known public ranking (a75 69.24 < chunked_dpt 66.28).
# If REAL-FID ranks a75 < chunk, we finally have a gate that predicts public.
cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=/workspace/experiments/wan_ticft
REAL_ROOT=${REAL_ROOT:-$ROOT/frames_balanced_val_125/val}
WORK=${WORK:-$ROOT/real_fid_work}
CUDA_ID="${CUDA_VISIBLE_DEVICES:-2}"; CUDA_ID="${CUDA_ID%%,*}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

A75=$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/videos
CHUNK_FIX=$ROOT/bdd_depth_chunkbase_sweep_dpt_eval24/depth_layered_a75/videos
CHUNK_ADA=$ROOT/bdd_depth_chunkbase_sweep_dpt_eval24/depth_layered_adaptive_a75/videos
CHUNK_A00=$ROOT/bdd_depth_chunkbase_sweep_dpt_eval24/depth_layered_a00/videos

echo "=== Trustworthy REAL-FID validation (val BDD vs real GT) ==="
$PYTHON -c "import pytorch_fid" 2>/dev/null || { echo "installing pytorch-fid..."; $PYTHON -m pip install --quiet pytorch-fid; }

ARGS=()
[[ -d "$A75" ]] && ARGS+=( --candidate "a75=$A75" )
[[ -d "$CHUNK_FIX" ]] && ARGS+=( --candidate "chunk_fixed=$CHUNK_FIX" )
[[ -d "$CHUNK_ADA" ]] && ARGS+=( --candidate "chunk_adaptive=$CHUNK_ADA" )
[[ -d "$CHUNK_A00" ]] && ARGS+=( --candidate "chunk_a00=$CHUNK_A00" )

"$PYTHON" -m wan_ticft.evaluation.compute_real_fid \
  --real_root "$REAL_ROOT" --work "$WORK" --limit 24 "${ARGS[@]}"

echo
echo "INTERPRET: public says a75(69.24) better than chunked_dpt(66.28) => a75 REAL-FID must be LOWER."
echo "  If a75 < chunk_fixed here -> gate is TRUSTWORTHY -> use it to filter all future experiments offline."
echo "  If not -> official FID differs from val Inception-FID; investigate (val vs test domain)."
