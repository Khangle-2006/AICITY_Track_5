#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
INDEX=${INDEX:-/workspace/experiments/index}
TEST_CSV=${TEST_CSV:-$INDEX/test_index.csv}
OUT=${OUT:-$ROOT/ltx2b_chunked_traffic_bdd_test}
LTX_REPO=${LTX_REPO:-/workspace/third_party/LTX-Video}
LTX_PYTHON=${LTX_PYTHON:-$ROOT/ltxv_env/bin/python}
PIPELINE_CONFIG=${PIPELINE_CONFIG:-/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml}
LIMIT=${LIMIT:-0}
OFFSET=${OFFSET:-0}
CHUNK_FRAMES=${CHUNK_FRAMES:-16}
PROMPT_MODE=${PROMPT_MODE:-traffic_detail}
CUDA_ID="${CUDA_VISIBLE_DEVICES:-2}"
CUDA_ID="${CUDA_ID%%,*}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== LTX 2B chunked traffic-detail BDD test ==="
echo "test_csv=$TEST_CSV"
echo "out=$OUT"
echo "limit=$LIMIT offset=$OFFSET chunk_frames=$CHUNK_FRAMES prompt_mode=$PROMPT_MODE"

for path in "$TEST_CSV" "$PIPELINE_CONFIG"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$CUDA_ID" 12000

"$PYTHON" -m wan_ticft.evaluation.run_ltxv_test_bdd_chunked \
  --test_csv "$TEST_CSV" \
  --out_dir "$OUT" \
  --ltx_repo "$LTX_REPO" \
  --ltx_python "$LTX_PYTHON" \
  --pipeline_config "$PIPELINE_CONFIG" \
  --limit "$LIMIT" \
  --offset "$OFFSET" \
  --max_history 41 \
  --chunk_frames "$CHUNK_FRAMES" \
  --height 720 \
  --width 1280 \
  --prompt_mode "$PROMPT_MODE" \
  --gpu_id "$CUDA_ID" \
  --min_gpu_free_mib 18000 \
  --chunk_retries 3 \
  --retry_sleep 90 \
  --resume

echo
echo "BDD chunked test report:"
echo "  $OUT/report.csv"
