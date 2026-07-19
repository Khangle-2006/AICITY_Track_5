#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
LIMIT_PER_DOMAIN=${LIMIT_PER_DOMAIN:-45}
TOTAL_ROWS=$((LIMIT_PER_DOMAIN * 2))
OUT=${OUT:-$ROOT/wts_object_layer_guarded_full${TOTAL_ROWS}}
ROUTE_DIR=${ROUTE_DIR:-$OUT/routed_reports}
MANIFEST=${MANIFEST:-$ROOT/manifests/val_ticft_balanced_125_frames.jsonl}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Guarded WTS object-layer robustness gate: ${LIMIT_PER_DOMAIN} samples/domain ==="
echo "Routes overhead actor-layer only when actor mask stats are small enough."

rm -rf "$OUT"
mkdir -p "$ROUTE_DIR"

echo
echo "[1/5] Materialize current components and actor-layer with mask stats"
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
echo "[2/5] Build current and guarded reports"
"$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
  --overhead_report "$OUT/region_norm_long24_report.csv" \
  --vehicle_report "$OUT/region_norm_medium_report.csv" \
  --route_label current_overhead_long24_vehicle_medium \
  --out_report "$ROUTE_DIR/current_report.csv"

"$PYTHON" -m wan_ticft.evaluation.build_wts_actor_guarded_report \
  --current_report "$ROUTE_DIR/current_report.csv" \
  --actor_report "$OUT/actor_layer_small_filled_report.csv" \
  --route_label guard005 \
  --max_sum_frac 0.005 \
  --max_mean 0.006 \
  --max_max_frac 0.012 \
  --max_components 40 \
  --max_raw_sum_frac 0.25 \
  --out_report "$ROUTE_DIR/guard005_report.csv"
"$PYTHON" -m wan_ticft.evaluation.build_wts_actor_guarded_report \
  --current_report "$ROUTE_DIR/current_report.csv" \
  --actor_report "$OUT/actor_layer_small_filled_report.csv" \
  --route_label guard010 \
  --max_sum_frac 0.010 \
  --max_mean 0.012 \
  --max_max_frac 0.018 \
  --max_components 60 \
  --max_raw_sum_frac 0.30 \
  --out_report "$ROUTE_DIR/guard010_report.csv"
"$PYTHON" -m wan_ticft.evaluation.build_wts_actor_guarded_report \
  --current_report "$ROUTE_DIR/current_report.csv" \
  --actor_report "$OUT/actor_layer_small_filled_report.csv" \
  --route_label guard015 \
  --max_sum_frac 0.015 \
  --max_mean 0.018 \
  --max_max_frac 0.022 \
  --max_components 80 \
  --max_raw_sum_frac 0.35 \
  --out_report "$ROUTE_DIR/guard015_report.csv"

echo
echo "[3/5] Guarded pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_report_pixels \
  --target_report "$OUT/target_report.csv" \
  --report "current=$ROUTE_DIR/current_report.csv" \
  --report "guard005=$ROUTE_DIR/guard005_report.csv" \
  --report "guard010=$ROUTE_DIR/guard010_report.csv" \
  --report "guard015=$ROUTE_DIR/guard015_report.csv" \
  --out_csv "$ROOT/wts_object_layer_guarded_full${TOTAL_ROWS}_pixel_gate.csv"

echo
echo "[4/5] Guarded relative FID/FVD proxy"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --target_report "$OUT/target_report.csv" \
  --full_video \
  --report "current=$ROUTE_DIR/current_report.csv" \
  --report "guard005=$ROUTE_DIR/guard005_report.csv" \
  --report "guard010=$ROUTE_DIR/guard010_report.csv" \
  --report "guard015=$ROUTE_DIR/guard015_report.csv" \
  --out_csv "$ROOT/wts_object_layer_guarded_full${TOTAL_ROWS}_proxy_gate.csv"

echo
echo "[5/5] Selected actor counts"
for report in "$ROUTE_DIR"/guard*_report.csv; do
  echo "$(basename "$report")"
  "$PYTHON" - <<PY
import csv
from pathlib import Path
path = Path("$report")
rows = list(csv.DictReader(path.open(encoding="utf-8")))
print("selected_actor", sum(row.get("guard_selected_actor") == "1" for row in rows), "rows", len(rows))
PY
done

echo
echo "Guarded promotion rule:"
echo "  - Promote only if a guard improves FVD versus current,"
echo "    does not worsen FID by >0.03, and loses <0.04 PSNR / <0.0015 SSIM."
echo "  - If none pass, current remains safer than object-layer routing."
