#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
EVAL_DEPTH="$ROOT/bdd_supervised_ltx_depth_eval24/depth"
EVAL_BASE="$ROOT/ltxv_bdd25_gate"
VISTA_REPO="${VISTA_REPO:-/workspace/third_party/Vista}"
VISTA_PY="${VISTA_PY:-/workspace/experiments/wan_ticft/vista_env/bin/python}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"
LIMIT="${LIMIT:-4}"
N_STEPS="${N_STEPS:-30}"
OUT="$ROOT/vista_img_bdd${LIMIT}_s${N_STEPS}"

echo "=== Vista IMG BDD gate ==="
echo "This is a last-frame-anchor feasibility smoke, not final integration."
echo "out=$OUT"
echo "limit=$LIMIT n_steps=$N_STEPS"

for path in "$MANIFEST" "$EVAL_BASE/report.csv" "$EVAL_DEPTH/depth_layered_a75/report.csv" "$VISTA_REPO/sample.py" "$VISTA_REPO/ckpts/vista.safetensors" "$VISTA_PY"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TOKENIZERS_PARALLELISM=false \
"$PYTHON" -m wan_ticft.evaluation.run_vista_img_bdd_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --vista_repo "$VISTA_REPO" \
  --vista_python "$VISTA_PY" \
  --limit "$LIMIT" \
  --view_type bdd \
  --n_rounds 2 \
  --n_frames 25 \
  --n_steps "$N_STEPS" \
  --height 576 \
  --width 1024 \
  --low_vram \
  --reuse_raw \
  --resume

echo "[1/3] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    vista_img="$OUT/report.csv" \
    ltx2b_base="$EVAL_BASE/report.csv" \
    depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo "[2/3] FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report vista_img="$OUT/report.csv" \
  --report ltx2b_base="$EVAL_BASE/report.csv" \
  --report depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
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
      vista_img="$OUT/report.csv" \
      ltx2b_base="$EVAL_BASE/report.csv" \
      depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
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
echo "  Continue only if vista_img preserves scene/road/actors and materially improves FID/FVD proxy."
echo "  If pixel collapses and FID/FVD is not clearly better, close raw Vista IMG and move to VSR/detailer."
