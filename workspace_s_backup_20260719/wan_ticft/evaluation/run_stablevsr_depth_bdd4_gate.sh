#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
EVAL_DEPTH="$ROOT/bdd_supervised_ltx_depth_eval24/depth"
EVAL_BASE="$ROOT/ltxv_bdd25_gate"
DEPTH_REPORT="$EVAL_DEPTH/depth_layered_a75/report.csv"
STABLEVSR_REPO="${STABLEVSR_REPO:-/workspace/third_party/StableVSR}"
STABLEVSR_PY="${STABLEVSR_PY:-/workspace/experiments/wan_ticft/stablevsr_env/bin/python}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"
LIMIT="${LIMIT:-4}"
STEPS="${STEPS:-20}"
LR_WIDTH="${LR_WIDTH:-320}"
LR_HEIGHT="${LR_HEIGHT:-180}"
OUT="$ROOT/stablevsr_depth_a75_bdd${LIMIT}_${LR_WIDTH}x${LR_HEIGHT}_s${STEPS}"

echo "=== StableVSR depth_a75 BDD gate ==="
echo "out=$OUT"
echo "limit=$LIMIT steps=$STEPS lr=${LR_WIDTH}x${LR_HEIGHT}"

for path in "$MANIFEST" "$DEPTH_REPORT" "$EVAL_BASE/report.csv" "$STABLEVSR_REPO/test.py" "$STABLEVSR_PY"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TOKENIZERS_PARALLELISM=false \
"$PYTHON" -m wan_ticft.evaluation.run_stablevsr_depth_gate \
  --manifest "$MANIFEST" \
  --base_report "$DEPTH_REPORT" \
  --out_dir "$OUT" \
  --stablevsr_repo "$STABLEVSR_REPO" \
  --stablevsr_python "$STABLEVSR_PY" \
  --limit "$LIMIT" \
  --view_type bdd \
  --num_inference_steps "$STEPS" \
  --lr_width "$LR_WIDTH" \
  --lr_height "$LR_HEIGHT" \
  --reuse_sr

echo "[1/3] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    stablevsr="$OUT/report.csv" \
    depth_a75="$DEPTH_REPORT" \
    ltx2b_base="$EVAL_BASE/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo "[2/3] FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report stablevsr="$OUT/report.csv" \
  --report depth_a75="$DEPTH_REPORT" \
  --report ltx2b_base="$EVAL_BASE/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 8 \
  --frame_stride 1

echo "[3/3] Contact sheets"
mapfile -t GENERATED_SAMPLE_IDS < <("$PYTHON" - <<PY
import json
for line in open("$OUT/selected_manifest.jsonl", encoding="utf-8"):
    if line.strip():
        print(json.loads(line)["sample_id"])
PY
)
for sample_id in "${GENERATED_SAMPLE_IDS[@]}"; do
  "$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
    --manifest "$OUT/selected_manifest.jsonl" \
    --reports \
      stablevsr="$OUT/report.csv" \
      depth_a75="$DEPTH_REPORT" \
      ltx2b_base="$EVAL_BASE/report.csv" \
    --sample_id "$sample_id" \
    --out_dir "${OUT}_compare_sheets" \
    --frames 8 \
    --tile_width 260
done

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_compare_sheets"
echo
echo "Gate:"
echo "  Continue only if stablevsr improves FID/LPIPS-like visual quality without severe FVD flicker or pixel collapse."
