#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl}
OUT=${OUT:-$ROOT/ltx2b_chunked_traffic_bdd24_gate}
LTX_REPO=${LTX_REPO:-/workspace/third_party/LTX-Video}
LTX_PYTHON=${LTX_PYTHON:-$ROOT/ltxv_env/bin/python}
PIPELINE_CONFIG=${PIPELINE_CONFIG:-/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml}
DEPTH_REPORT=${DEPTH_REPORT:-$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv}
LIMIT=${LIMIT:-24}
CHUNK_FRAMES=${CHUNK_FRAMES:-16}
PROMPT_MODE=${PROMPT_MODE:-traffic_detail}
CUDA_ID="${CUDA_VISIBLE_DEVICES:-2}"
CUDA_ID="${CUDA_ID%%,*}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== LTX 2B chunked traffic-detail BDD24 gate ==="
echo "manifest=$MANIFEST"
echo "out=$OUT"
echo "limit=$LIMIT chunk_frames=$CHUNK_FRAMES prompt_mode=$PROMPT_MODE"

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$CUDA_ID" 12000

echo "[1/4] Generate chunked LTX videos"
"$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_chunked_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --ltx_repo "$LTX_REPO" \
  --ltx_python "$LTX_PYTHON" \
  --pipeline_config "$PIPELINE_CONFIG" \
  --limit "$LIMIT" \
  --view_type bdd \
  --prompt_mode "$PROMPT_MODE" \
  --height 704 \
  --width 1216 \
  --chunk_frames "$CHUNK_FRAMES" \
  --skip_bad

echo "[2/4] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports ltx2b_chunked="$OUT/report.csv" depth_a75="$DEPTH_REPORT" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "$OUT/eval_pixel.csv"

echo "[3/4] FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$CUDA_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report ltx2b_chunked="$OUT/report.csv" \
  --report depth_a75="$DEPTH_REPORT" \
  --clip_frames 8 \
  --frame_stride 1 \
  --out_csv "$OUT/eval_proxy.csv"

echo "[4/4] Contact sheets"
mapfile -t SAMPLE_IDS < <("$PYTHON" - <<PY
import json
keep = {0, 5, 9, 13, 20, 23}
for i, line in enumerate(open("$OUT/selected_manifest.jsonl", encoding="utf-8")):
    if i in keep and line.strip():
        print(json.loads(line)["sample_id"])
PY
)
for sample_id in "${SAMPLE_IDS[@]}"; do
  "$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
    --manifest "$OUT/selected_manifest.jsonl" \
    --reports ltx2b_chunked="$OUT/report.csv" depth_a75="$DEPTH_REPORT" \
    --sample_id "$sample_id" \
    --out_dir "$OUT/compare_sheets" \
    --frames 8 \
    --tile_width 220
done

echo
echo "report=$OUT/report.csv"
echo "pixel=$OUT/eval_pixel.csv"
echo "proxy=$OUT/eval_proxy.csv"
echo "sheets=$OUT/compare_sheets"
echo
echo "Gate:"
echo "  Promote only if ltx2b_chunked beats depth_a75 on BDD24 pixel and proxy,"
echo "  and sheets do not show hallucinated road, vehicles, or geometry drift."
