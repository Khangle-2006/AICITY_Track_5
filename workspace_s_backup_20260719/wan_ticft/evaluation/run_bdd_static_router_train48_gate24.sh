#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"
TRAIN_LIMIT="${TRAIN_LIMIT:-48}"
MARGIN="${MARGIN:-0.25}"
THRESH="${THRESH:-0.50}"

TRAIN_MANIFEST="$ROOT/bdd_supervised_ltx_depth_train${TRAIN_LIMIT}/ltx_base/selected_manifest_repaired_conditions.jsonl"
TRAIN_DEPTH="$ROOT/bdd_supervised_ltx_depth_train${TRAIN_LIMIT}/depth/depth_layered_a75/report.csv"
TRAIN_REPEAT="$ROOT/bdd_supervised_repeat_depth_train${TRAIN_LIMIT}/repeat_last/report.csv"
EVAL_MANIFEST="$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl"
EVAL_DEPTH="$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv"
EVAL_REPEAT="$ROOT/bdd_supervised_repeat_depth_eval24/repeat_last/report.csv"
PHASE12="$ROOT/bdd24_phase12_convex_depth_rule/report.csv"
OUT="$ROOT/bdd_static_router_train${TRAIN_LIMIT}_m${MARGIN}_t${THRESH}_gate24"

echo "=== BDD static learned router train${TRAIN_LIMIT} -> BDD24 ==="
echo "margin=$MARGIN threshold=$THRESH"
echo "out=$OUT"

for path in "$TRAIN_MANIFEST" "$TRAIN_DEPTH" "$TRAIN_REPEAT" "$EVAL_MANIFEST" "$EVAL_DEPTH" "$EVAL_REPEAT"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

echo "[1/4] Train/infer static router"
"$PYTHON" -m wan_ticft.evaluation.train_bdd_static_depth_router \
  --train_manifest "$TRAIN_MANIFEST" \
  --train_depth_report "$TRAIN_DEPTH" \
  --train_repeat_report "$TRAIN_REPEAT" \
  --eval_manifest "$EVAL_MANIFEST" \
  --eval_depth_report "$EVAL_DEPTH" \
  --eval_repeat_report "$EVAL_REPEAT" \
  --out_dir "$OUT" \
  --train_limit "$TRAIN_LIMIT" \
  --repeat_margin "$MARGIN" \
  --repeat_threshold "$THRESH" \
  --clean

echo "[2/4] Pixel metrics"
REPORTS=(depth_a75="$EVAL_DEPTH" static_router="$OUT/routed_report.csv" repeat="$EVAL_REPEAT")
if [[ -s "$PHASE12" ]]; then
  REPORTS+=(phase12="$PHASE12")
fi
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_MANIFEST" \
  --reports "${REPORTS[@]}" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo "[3/4] FID/FVD proxy"
PROXY_REPORTS=(--report depth_a75="$EVAL_DEPTH" --report static_router="$OUT/routed_report.csv" --report repeat="$EVAL_REPEAT")
if [[ -s "$PHASE12" ]]; then
  PROXY_REPORTS+=(--report phase12="$PHASE12")
fi
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_MANIFEST" \
  "${PROXY_REPORTS[@]}" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 8 \
  --frame_stride 1

echo "[4/4] Contact sheets"
mapfile -t SAMPLE_IDS < <("$PYTHON" - <<PY
import json
for i, line in enumerate(open("$EVAL_MANIFEST", encoding="utf-8")):
    if i in {0, 5, 9, 13, 20, 23} and line.strip():
        print(json.loads(line)["sample_id"])
PY
)
for sample_id in "${SAMPLE_IDS[@]}"; do
  "$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
    --manifest "$EVAL_MANIFEST" \
    --reports "${REPORTS[@]}" \
    --sample_id "$sample_id" \
    --out_dir "${OUT}_compare_sheets" \
    --frames 8 \
    --tile_width 220
done

echo
echo "choices=$OUT/choices.csv"
echo "train_labels=$OUT/train_labels.csv"
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_compare_sheets"
echo
echo "Gate:"
echo "  Continue only if static_router beats phase12 or gives a safer FVD/FID tradeoff."
