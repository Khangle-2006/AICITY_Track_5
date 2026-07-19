#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl}
OUT=${OUT:-$ROOT/ltx2b_traffic_prompt_bdd4_gate}
LTX_REPO=${LTX_REPO:-/workspace/third_party/LTX-Video}
LTX_PYTHON=${LTX_PYTHON:-$ROOT/ltxv_env/bin/python}
PIPELINE_CONFIG=${PIPELINE_CONFIG:-/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml}
DEPTH_REPORT=${DEPTH_REPORT:-$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv}
LIMIT=${LIMIT:-4}
CUDA_ID="${CUDA_VISIBLE_DEVICES:-2}"
CUDA_ID="${CUDA_ID%%,*}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== LTX 2B traffic-detail prompt BDD gate ==="
echo "manifest=$MANIFEST"
echo "out=$OUT"
echo "limit=$LIMIT"
echo "prompt_mode=traffic_detail"

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$CUDA_ID" 12000

"$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --ltx_repo "$LTX_REPO" \
  --ltx_python "$LTX_PYTHON" \
  --pipeline_config "$PIPELINE_CONFIG" \
  --limit "$LIMIT" \
  --view_type bdd \
  --prompt_mode traffic_detail \
  --height 704 \
  --width 1216 \
  --num_frames 121 \
  --skip_bad

echo "[1/3] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports ltx2b_traffic="$OUT/report.csv" depth_a75="$DEPTH_REPORT" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "$OUT/eval_pixel.csv"

echo "[2/3] FID/FVD proxy"
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report ltx2b_traffic="$OUT/report.csv" \
  --report depth_a75="$DEPTH_REPORT" \
  --clip_frames 8 \
  --out_csv "$OUT/eval_proxy.csv"

echo "[3/3] Contact sheets"
SHEETS="$OUT/compare_sheets"
mkdir -p "$SHEETS"
mapfile -t SAMPLE_IDS < <("$PYTHON" - <<PY
import json
from pathlib import Path
path = Path("$OUT/selected_manifest.jsonl")
for line in path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        print(json.loads(line)["sample_id"])
PY
)
SAMPLE_ARGS=()
for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
  SAMPLE_ARGS+=(--sample_id "$SAMPLE_ID")
done
"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports depth_a75="$DEPTH_REPORT" ltx2b_traffic="$OUT/report.csv" \
  "${SAMPLE_ARGS[@]}" \
  --out_dir "$SHEETS" \
  --frames 8 || true

echo
echo "report=$OUT/report.csv"
echo "pixel=$OUT/eval_pixel.csv"
echo "proxy=$OUT/eval_proxy.csv"
echo "sheets=$SHEETS"
