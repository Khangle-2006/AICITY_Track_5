#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
OUT=${OUT:-$ROOT/wts_object_layer_gate20}
ROUTE_DIR=${ROUTE_DIR:-$OUT/routed_reports}
MANIFEST=${MANIFEST:-$ROOT/manifests/val_ticft_balanced_125_frames.jsonl}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== WTS object-layer motion gate: not dense-flow-only ==="
echo "General rule: uses only input-frame background/motion statistics, no sample IDs."

rm -rf "$OUT"
mkdir -p "$ROUTE_DIR"

echo
echo "[1/4] Materialize object-layer candidates on balanced WTS validation"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --limit_per_domain 10 \
  --frames 50 \
  --candidates \
    repeat_last \
    region_norm_medium \
    region_norm_long24 \
    actor_layer_median \
    actor_layer_filled \
  --clean

echo
echo "[2/4] Direct domain/macro FID-FVD proxy"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --target_report "$OUT/target_report.csv" \
  --full_video \
  --report "repeat_last=$OUT/repeat_last_report.csv" \
  --report "region_medium=$OUT/region_norm_medium_report.csv" \
  --report "region_long24=$OUT/region_norm_long24_report.csv" \
  --report "actor_layer_median=$OUT/actor_layer_median_report.csv" \
  --report "actor_layer_filled=$OUT/actor_layer_filled_report.csv" \
  --out_csv "$ROOT/wts_object_layer_distribution_proxy_gate20.csv"

echo
echo "[3/4] Routed candidates: keep current best WTS branch where object-layer loses"
"$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
  --overhead_report "$OUT/region_norm_long24_report.csv" \
  --vehicle_report "$OUT/region_norm_medium_report.csv" \
  --route_label current_overhead_long24_vehicle_medium \
  --out_report "$ROUTE_DIR/current_report.csv"
"$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
  --overhead_report "$OUT/actor_layer_median_report.csv" \
  --vehicle_report "$OUT/region_norm_medium_report.csv" \
  --route_label overhead_actor_layer_median_vehicle_medium \
  --out_report "$ROUTE_DIR/overhead_actor_layer_median_vehicle_medium_report.csv"
"$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
  --overhead_report "$OUT/actor_layer_filled_report.csv" \
  --vehicle_report "$OUT/region_norm_medium_report.csv" \
  --route_label overhead_actor_layer_filled_vehicle_medium \
  --out_report "$ROUTE_DIR/overhead_actor_layer_filled_vehicle_medium_report.csv"
"$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
  --overhead_report "$OUT/region_norm_long24_report.csv" \
  --vehicle_report "$OUT/actor_layer_median_report.csv" \
  --route_label overhead_long24_vehicle_actor_layer_median \
  --out_report "$ROUTE_DIR/overhead_long24_vehicle_actor_layer_median_report.csv"
"$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
  --overhead_report "$OUT/region_norm_long24_report.csv" \
  --vehicle_report "$OUT/actor_layer_filled_report.csv" \
  --route_label overhead_long24_vehicle_actor_layer_filled \
  --out_report "$ROUTE_DIR/overhead_long24_vehicle_actor_layer_filled_report.csv"

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --target_report "$OUT/target_report.csv" \
  --full_video \
  --report "current=$ROUTE_DIR/current_report.csv" \
  --report "overhead_actor_layer_median=$ROUTE_DIR/overhead_actor_layer_median_vehicle_medium_report.csv" \
  --report "overhead_actor_layer_filled=$ROUTE_DIR/overhead_actor_layer_filled_vehicle_medium_report.csv" \
  --report "vehicle_actor_layer_median=$ROUTE_DIR/overhead_long24_vehicle_actor_layer_median_report.csv" \
  --report "vehicle_actor_layer_filled=$ROUTE_DIR/overhead_long24_vehicle_actor_layer_filled_report.csv" \
  --out_csv "$ROOT/wts_object_layer_routed_proxy_gate20.csv"

echo
echo "[4/4] Contact sheets for visual QA"
"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/actor_layer_median" \
  --mode videos \
  --out_dir "$OUT/actor_layer_median_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4
"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/actor_layer_filled" \
  --mode videos \
  --out_dir "$OUT/actor_layer_filled_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

echo
echo "Gate rule:"
echo "  - Promote only if a routed object-layer candidate improves both FID and FVD"
echo "    proxies versus current."
echo "  - Visual QA must show less small-object smear without cut-paste halos or"
echo "    duplicated/stuck actors."
echo "  - If fail, close non-generative object-layer WTS motion for now."
