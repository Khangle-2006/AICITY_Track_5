#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"

EVAL_BASE="$ROOT/ltxv_bdd25_gate"
EVAL_DEPTH="$ROOT/bdd_supervised_ltx_depth_eval24/depth"
EVAL_REPEAT="$ROOT/bdd_supervised_repeat_depth_eval24/repeat_last"
OUT="$ROOT/bdd_motion_persistence_router_bdd24"

echo "=== BDD motion-persistence router gate24 ==="
echo "Route to repeat-last only when history and depth prediction are both low-motion."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 4000

for path in \
  "$EVAL_BASE/selected_manifest.jsonl" \
  "$EVAL_DEPTH/depth_layered_a75/report.csv"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

echo "[1/5] Ensure eval repeat-last videos"
if [[ -s "$EVAL_REPEAT/report.csv" ]]; then
  echo "  found existing repeat report"
else
  "$PYTHON" -m wan_ticft.evaluation.materialize_repeat_last_videos \
    --manifest "$EVAL_BASE/selected_manifest.jsonl" \
    --reference_report "$EVAL_DEPTH/depth_layered_a75/report.csv" \
    --out_dir "$EVAL_REPEAT" \
    --view_type bdd \
    --limit 24
fi

echo "[2/5] Build motion-threshold route reports"
"$PYTHON" -m wan_ticft.evaluation.materialize_motion_persistence_router \
  --manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --depth_report "$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --repeat_report "$EVAL_REPEAT/report.csv" \
  --out_root "$OUT" \
  --view_type bdd \
  --limit 24 \
  --hist_thresholds 1.0 2.0 3.0 4.0 \
  --pred_thresholds 0.15 0.30 0.60 1.00 \
  --motion_frames 12 \
  --clean

echo "[3/5] Pixel score all route variants"
REPORT_ARGS=(
  repeat_video="$EVAL_REPEAT/report.csv"
  depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv"
)
for report in "$OUT"/route_*/report.csv; do
  label="$(basename "$(dirname "$report")")"
  REPORT_ARGS+=("$label=$report")
done

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --reports "${REPORT_ARGS[@]}" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo "[4/5] FID/FVD proxy for conservative shortlist"
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --report depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --report repeat_video="$EVAL_REPEAT/report.csv" \
  --report route_h1_p0p15="$OUT/route_h1_p0p15/report.csv" \
  --report route_h2_p0p3="$OUT/route_h2_p0p3/report.csv" \
  --report route_h3_p0p6="$OUT/route_h3_p0p6/report.csv" \
  --report route_h4_p1="$OUT/route_h4_p1/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 8 \
  --frame_stride 1

echo "[5/5] Contact sheets for shortlisted routes"
mapfile -t SAMPLE_IDS < <("$PYTHON" - <<PY
import json
for i, line in enumerate(open("$EVAL_BASE/selected_manifest.jsonl", encoding="utf-8")):
    if i in {0, 6, 9, 10, 14, 19, 23} and line.strip():
        print(json.loads(line)["sample_id"])
PY
)
for sample_id in "${SAMPLE_IDS[@]}"; do
  "$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
    --manifest "$EVAL_BASE/selected_manifest.jsonl" \
    --reports \
      depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
      repeat_video="$EVAL_REPEAT/report.csv" \
      route_h2_p0p3="$OUT/route_h2_p0p3/report.csv" \
      route_h3_p0p6="$OUT/route_h3_p0p6/report.csv" \
    --sample_id "$sample_id" \
    --out_dir "${OUT}_compare_sheets" \
    --frames 8 \
    --tile_width 220
done

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "diagnostics=$OUT/diagnostics.csv"
echo "sheets=${OUT}_compare_sheets"
echo
echo "Gate:"
echo "  Continue only if a route improves PSNR/AbsDiff and keeps FVD close to depth_a75."
echo "  If all repeat routes worsen FVD materially, keep BDD depth_a75/a125 and close persistence."
