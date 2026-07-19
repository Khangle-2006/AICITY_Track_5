#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/ltx2b_chunked_traffic_bdd24_gate/selected_manifest.jsonl}
BASE_REPORT=${BASE_REPORT:-$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv}
DETAIL_REPORT=${DETAIL_REPORT:-$ROOT/ltx2b_chunked_traffic_bdd24_gate/report.csv}
OUT=${OUT:-$ROOT/ltx2b_chunked_realism_repair_bdd24_gate}
ALPHAS=${ALPHAS:-"0.08 0.12 0.16 0.20"}
SIGMA=${SIGMA:-1.2}
RESIDUAL_CLIP=${RESIDUAL_CLIP:-18}
CUDA_ID="${CUDA_VISIBLE_DEVICES:-2}"
CUDA_ID="${CUDA_ID%%,*}"

echo "=== LTX2B chunked realism repair BDD24 gate ==="
echo "manifest=$MANIFEST"
echo "base_report=$BASE_REPORT"
echo "detail_report=$DETAIL_REPORT"
echo "out=$OUT"
echo "alphas=$ALPHAS sigma=$SIGMA residual_clip=$RESIDUAL_CLIP"

for path in "$MANIFEST" "$BASE_REPORT" "$DETAIL_REPORT"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

rm -rf "$OUT"

ALPHA_ARGS=()
REPORT_ARGS=(depth_a75="$BASE_REPORT" ltx2b_chunked="$DETAIL_REPORT")
PROXY_ARGS=(--report depth_a75="$BASE_REPORT" --report ltx2b_chunked="$DETAIL_REPORT")
FULL_ARGS=(depth_a75="$BASE_REPORT" ltx2b_chunked="$DETAIL_REPORT")
for alpha in $ALPHAS; do
  ALPHA_ARGS+=(--alpha "$alpha")
  tag=$(printf "realism_a%02d" "$(python - <<PY
print(round(float("$alpha") * 100))
PY
)")
  REPORT_ARGS+=("$tag=$OUT/$tag/report.csv")
  PROXY_ARGS+=(--report "$tag=$OUT/$tag/report.csv")
  FULL_ARGS+=("$tag=$OUT/$tag/report.csv")
done

echo "[1/5] Materialize high-frequency realism repairs"
"$PYTHON" -m wan_ticft.evaluation.materialize_ltx_detail_realism_repair \
  --base_report "$BASE_REPORT" \
  --detail_report "$DETAIL_REPORT" \
  --out_dir "$OUT" \
  --label_prefix realism \
  --sigma "$SIGMA" \
  --residual_clip "$RESIDUAL_CLIP" \
  --color_match frame \
  "${ALPHA_ARGS[@]}"

echo "[2/5] Sparse target pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports "${REPORT_ARGS[@]}" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "$OUT/eval_sparse_pixel.csv"

echo "[3/5] Full-horizon pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports_full_horizon \
  --manifest "$MANIFEST" \
  --reports "${FULL_ARGS[@]}" \
  --include_repeat_last \
  --out_csv "$OUT/eval_full_horizon.csv"

echo "[4/5] FID/FVD proxy"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$CUDA_ID" 12000
CUDA_VISIBLE_DEVICES="$CUDA_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  "${PROXY_ARGS[@]}" \
  --clip_frames 8 \
  --frame_stride 1 \
  --out_csv "$OUT/eval_proxy.csv"

echo "[5/5] Contact sheets"
"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$MANIFEST" \
  --reports "${REPORT_ARGS[@]}" \
  --sample_id val_bdd_pc_5k_video7040_phase0 \
  --sample_id val_bdd_pc_5k_video2556_phase2 \
  --sample_id val_bdd_pc_5k_video8139_phase2 \
  --sample_id val_bdd_pc_5k_video8170_phase4 \
  --out_dir "$OUT/compare_sheets" \
  --frames 8 \
  --tile_width 220

echo
echo "sparse=$OUT/eval_sparse_pixel.csv"
echo "full_horizon=$OUT/eval_full_horizon.csv"
echo "proxy=$OUT/eval_proxy.csv"
echo "sheets=$OUT/compare_sheets"
echo
echo "Gate:"
echo "  Promote only if a small-alpha repair improves visual realness/FID without losing the chunk sharpness benefit."
