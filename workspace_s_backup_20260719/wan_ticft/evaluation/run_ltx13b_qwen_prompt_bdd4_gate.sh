#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl}
OUT=${OUT:-$ROOT/ltx13b_qwen_prompt_bdd4_gate}
PLANNER_OUT=${PLANNER_OUT:-$OUT/qwen_planner}
MODEL=${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
LIMIT=${LIMIT:-4}
MAX_PIXELS=${VLM_MAX_PIXELS:-401408}
DEPTH_REPORT=${DEPTH_REPORT:-$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv}
LTX_REPO=${LTX_REPO:-/workspace/third_party/LTX-Video}
PIPELINE_CONFIG=${PIPELINE_CONFIG:-$LTX_REPO/configs/ltxv-13b-0.9.8-distilled-noenhance.yaml}
HEIGHT=${HEIGHT:-720}
WIDTH=${WIDTH:-1280}
MAX_HISTORY_FRAMES=${MAX_HISTORY_FRAMES:-17}
NUM_FRAMES=${NUM_FRAMES:-49}
CUDA_ID="${CUDA_VISIBLE_DEVICES:-1}"
CUDA_ID="${CUDA_ID%%,*}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export FORCE_LTX_OFFLOAD="${FORCE_LTX_OFFLOAD:-1}"
export LTX_VAE_HW_TILING="${LTX_VAE_HW_TILING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Qwen2.5-VL prompt planner -> LTX 13B BDD gate ==="
echo "manifest=$MANIFEST"
echo "out=$OUT"
echo "planner_out=$PLANNER_OUT"
echo "limit=$LIMIT model=$MODEL size=${HEIGHT}x${WIDTH}"

for path in "$MANIFEST" "$DEPTH_REPORT" "$PIPELINE_CONFIG"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

rm -rf "$OUT"
mkdir -p "$OUT"

echo "[1/5] Qwen2.5-VL writes per-sample LTX prompts"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$CUDA_ID" 26000
CUDA_VISIBLE_DEVICES="$CUDA_ID" \
HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.run_qwen25vl_ltx_prompt_planner \
  --manifest "$MANIFEST" \
  --out_dir "$PLANNER_OUT" \
  --model "$MODEL" \
  --limit "$LIMIT" \
  --max_pixels "$MAX_PIXELS"

PLANNED_MANIFEST="$PLANNER_OUT/planned_manifest.jsonl"
[[ -s "$PLANNED_MANIFEST" ]] || { echo "Planner did not write $PLANNED_MANIFEST" >&2; exit 2; }

echo "[2/5] Generate LTX 13B with Qwen-planned prompts"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$CUDA_ID" 30000
"$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_gate \
  --manifest "$PLANNED_MANIFEST" \
  --out_dir "$OUT" \
  --ltx_repo "$LTX_REPO" \
  --ltx_python "$PYTHON" \
  --pipeline_config "$PIPELINE_CONFIG" \
  --limit "$LIMIT" \
  --view_type bdd \
  --max_history_frames "$MAX_HISTORY_FRAMES" \
  --prompt_mode manifest \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --skip_bad \
  --resume

echo "[3/5] Sparse target pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports ltx13b_qwen="$OUT/report.csv" depth_a75="$DEPTH_REPORT" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "$OUT/eval_pixel.csv"

echo "[4/5] Full-horizon pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports_full_horizon \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports ltx13b_qwen="$OUT/report.csv" depth_a75="$DEPTH_REPORT" \
  --include_repeat_last \
  --out_csv "$OUT/eval_full_horizon.csv"

echo "[5/5] FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$CUDA_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report ltx13b_qwen="$OUT/report.csv" \
  --report depth_a75="$DEPTH_REPORT" \
  --clip_frames 8 \
  --frame_stride 1 \
  --out_csv "$OUT/eval_proxy.csv"

echo
echo "planner=$PLANNER_OUT/planner_report.csv"
echo "report=$OUT/report.csv"
echo "pixel=$OUT/eval_pixel.csv"
echo "full_horizon=$OUT/eval_full_horizon.csv"
echo "proxy=$OUT/eval_proxy.csv"
echo
echo "Gate:"
echo "  Continue only if Qwen prompts improve 13B geometry/timing without losing its realism."
