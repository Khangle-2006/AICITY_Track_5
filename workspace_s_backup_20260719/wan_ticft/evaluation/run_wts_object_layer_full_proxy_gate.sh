#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
LIMIT_PER_DOMAIN=${LIMIT_PER_DOMAIN:-45}
TOTAL_ROWS=$((LIMIT_PER_DOMAIN * 2))
OUT=${OUT:-$ROOT/wts_object_layer_full${TOTAL_ROWS}}
ROUTE_DIR=${ROUTE_DIR:-$OUT/routed_reports}
MANIFEST=${MANIFEST:-$ROOT/manifests/val_ticft_balanced_125_frames.jsonl}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Full WTS object-layer robustness gate: ${LIMIT_PER_DOMAIN} samples/domain ==="
echo "This checks whether the overhead-small route generalizes beyond the 20-clip smoke."

rm -rf "$OUT"
mkdir -p "$ROUTE_DIR"

echo
echo "[1/4] Materialize current components and overhead small-object layer"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --limit_per_domain "$LIMIT_PER_DOMAIN" \
  --frames 50 \
  --candidates \
    region_norm_medium \
    region_norm_long24 \
    actor_layer_small_filled \
  --clean

echo
echo "[2/4] Build current and overhead-small routed reports"
"$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
  --overhead_report "$OUT/region_norm_long24_report.csv" \
  --vehicle_report "$OUT/region_norm_medium_report.csv" \
  --route_label current_overhead_long24_vehicle_medium \
  --out_report "$ROUTE_DIR/current_report.csv"
"$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
  --overhead_report "$OUT/actor_layer_small_filled_report.csv" \
  --vehicle_report "$OUT/region_norm_medium_report.csv" \
  --route_label overhead_actor_layer_small_filled_vehicle_medium \
  --out_report "$ROUTE_DIR/overhead_small_filled_report.csv"

echo
echo "[3/4] Full WTS pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_report_pixels \
  --target_report "$OUT/target_report.csv" \
  --report "current=$ROUTE_DIR/current_report.csv" \
  --report "overhead_small_filled=$ROUTE_DIR/overhead_small_filled_report.csv" \
  --out_csv "$ROOT/wts_object_layer_full${TOTAL_ROWS}_pixel_gate.csv"

echo
echo "[4/4] Full WTS relative FID/FVD proxy"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --target_report "$OUT/target_report.csv" \
  --full_video \
  --report "current=$ROUTE_DIR/current_report.csv" \
  --report "overhead_small_filled=$ROUTE_DIR/overhead_small_filled_report.csv" \
  --out_csv "$ROOT/wts_object_layer_full${TOTAL_ROWS}_proxy_gate.csv"

echo
echo "Full robustness rule:"
echo "  - Keep confidence in the submitted overhead-object candidate only if"
echo "    overhead_small_filled improves full-WTS FVD and does not materially hurt FID,"
echo "    PSNR, or SSIM versus current."
echo "  - If it fails full${TOTAL_ROWS}, treat the 20-clip smoke as overfit and prepare a"
echo "    fallback around the current best."
