#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl}
OUT=${OUT:-$ROOT/ltx13b_traffic_bdd4_gate}
LTX_REPO=${LTX_REPO:-/workspace/third_party/LTX-Video}
LTX_PYTHON=${LTX_PYTHON:-/opt/conda/envs/wan22/bin/python}
PIPELINE_CONFIG=${PIPELINE_CONFIG:-$LTX_REPO/configs/ltxv-13b-0.9.8-distilled-noenhance.yaml}
DEPTH_REPORT=${DEPTH_REPORT:-$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv}
LIMIT=${LIMIT:-4}
HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-768}
MAX_HISTORY_FRAMES=${MAX_HISTORY_FRAMES:-17}
NUM_FRAMES=${NUM_FRAMES:-49}
PROMPT_MODE=${PROMPT_MODE:-traffic_detail}
CUDA_ID="${CUDA_VISIBLE_DEVICES:-1}"
CUDA_ID="${CUDA_ID%%,*}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export FORCE_LTX_OFFLOAD="${FORCE_LTX_OFFLOAD:-1}"
export LTX_VAE_HW_TILING="${LTX_VAE_HW_TILING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== LTX 13B traffic-detail BDD gate ==="
echo "manifest=$MANIFEST"
echo "out=$OUT"
echo "limit=$LIMIT size=${HEIGHT}x${WIDTH} history=$MAX_HISTORY_FRAMES num_frames=$NUM_FRAMES prompt_mode=$PROMPT_MODE"
echo "config=$PIPELINE_CONFIG"

for path in "$MANIFEST" "$PIPELINE_CONFIG" "$DEPTH_REPORT"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$CUDA_ID" 30000

echo "[1/3] Generate LTX 13B candidates"
"$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --ltx_repo "$LTX_REPO" \
  --ltx_python "$LTX_PYTHON" \
  --pipeline_config "$PIPELINE_CONFIG" \
  --limit "$LIMIT" \
  --view_type bdd \
  --max_history_frames "$MAX_HISTORY_FRAMES" \
  --prompt_mode "$PROMPT_MODE" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --skip_bad \
  --resume

echo "[2/3] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports ltx13b="$OUT/report.csv" depth_a75="$DEPTH_REPORT" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "$OUT/eval_pixel.csv"

echo "[3/3] FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$CUDA_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report ltx13b="$OUT/report.csv" \
  --report depth_a75="$DEPTH_REPORT" \
  --clip_frames 8 \
  --frame_stride 1 \
  --out_csv "$OUT/eval_proxy.csv"

echo
echo "report=$OUT/report.csv"
echo "pixel=$OUT/eval_pixel.csv"
echo "proxy=$OUT/eval_proxy.csv"
echo
echo "Gate:"
echo "  Continue only if ltx13b improves FID/FVD or sheets look clearly better without geometry drift."
