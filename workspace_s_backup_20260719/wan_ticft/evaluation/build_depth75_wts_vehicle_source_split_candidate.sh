#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
TEST_CSV="$INDEX/test_index.csv"
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
BDD_RENDERED=${BDD_RENDERED:-$ROOT/depth_layered_a75_bdd_test}
WTS_BASE=${WTS_BASE:-$ROOT/wts_overhead_object_layer_vehicle_medium_test}
WTS_VEHICLE_RENDERED=${WTS_VEHICLE_RENDERED:-$ROOT/depth_layered_a75_wts_vehicle_test}
MODEL_CACHE=${MODEL_CACHE:-$ROOT/depth_layered_models}
DEPTH_ROUTE=${DEPTH_ROUTE:-normal}
ZIP=${ZIP:-}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

case "$DEPTH_ROUTE" in
  normal)
    SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_depth75_wts_vehicle_normal_only_candidate}
    REPORT=${REPORT:-$ROOT/hybrid_depth75_wts_vehicle_normal_only_candidate_report.csv}
    ;;
  event)
    SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_depth75_wts_vehicle_event_only_candidate}
    REPORT=${REPORT:-$ROOT/hybrid_depth75_wts_vehicle_event_only_candidate_report.csv}
    ;;
  all)
    SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_depth75_wts_vehicle_all_candidate}
    REPORT=${REPORT:-$ROOT/hybrid_depth75_wts_vehicle_all_candidate_report.csv}
    ;;
  *)
    echo "DEPTH_ROUTE must be one of: normal, event, all" >&2
    exit 2
    ;;
esac

echo "=== Build depth75 with WTS vehicle/IP source split ==="
echo "depth_route=$DEPTH_ROUTE"
if [[ -n "$ZIP" ]]; then
  echo "zip=$ZIP"
else
  echo "No zip is created."
fi

for path in "$TEST_CSV" "$BDD_RENDERED/report.csv"; do
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
[[ -d "$WTS_BASE" ]] || { echo "Missing WTS base: $WTS_BASE" >&2; exit 2; }

if [[ ! -f "$WTS_VEHICLE_RENDERED/report.csv" ]]; then
  echo "[1/3] Render depth-a75 for all WTS vehicle/IP test samples"
  /workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 12000
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  HF_HOME="$MODEL_CACHE/hf_home" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer_test \
    --test_csv "$TEST_CSV" \
    --base_dir "$WTS_BASE" \
    --out_dir "$WTS_VEHICLE_RENDERED" \
    --model_cache "$MODEL_CACHE/huggingface" \
    --view_type wts_vehicle_or_ip_camera \
    --alpha 0.75 \
    --focal_scale 0.9 \
    --point_stride 2 \
    --splat_radius 1 \
    --min_matches 20 \
    --min_inliers 12 \
    --photo_sigma 32 \
    --segment_batch 4 \
    --pose_smoothing_window 7 \
    --filled_pose_scale 0.35 \
    --confidence_ema 0.65 \
    --clean
else
  echo "[1/3] Reuse existing WTS vehicle/IP depth render: $WTS_VEHICLE_RENDERED"
fi

echo "[2/3] Build split-routed hybrid folder"
COMMON_ARGS=(
  --index_dir "$INDEX"
  --wan_dir "$BDD_RENDERED"
  --region_dir "$WTS_BASE"
  --out_dir "$SUBMISSION"
  --report_csv "$REPORT"
  --wan_label depth_layered_a75_bdd
  --region_label wts_overhead_object_layer_vehicle_medium
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera
  --clean
)
if [[ -n "$ZIP" ]]; then
  COMMON_ARGS+=(--zip "$ZIP")
fi

case "$DEPTH_ROUTE" in
  normal)
    "$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
      "${COMMON_ARGS[@]}" \
      --source_folder_override "wts_vehicle_or_ip_camera=normal=$WTS_VEHICLE_RENDERED=depth_a75_wts_vehicle_normal_only"
    ;;
  event)
    "$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
      "${COMMON_ARGS[@]}" \
      --view_override "wts_vehicle_or_ip_camera=$WTS_VEHICLE_RENDERED=depth_a75_wts_vehicle_event_default" \
      --source_folder_override "wts_vehicle_or_ip_camera=normal=$WTS_BASE=wts_vehicle_normal_base"
    ;;
  all)
    "$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
      "${COMMON_ARGS[@]}" \
      --view_override "wts_vehicle_or_ip_camera=$WTS_VEHICLE_RENDERED=depth_a75_wts_vehicle_all"
    ;;
esac

echo "[3/3] Verify complete submission folder"
"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$SUBMISSION" \
  --submission_dir "$SUBMISSION" \
  --verify_only

echo
echo "Route summary"
"$PYTHON" -m wan_ticft.evaluation.summarize_wts_vehicle_split \
  --test_csv "$TEST_CSV" \
  --report_csv "$REPORT"

echo
echo "Candidate verified:"
echo "  $SUBMISSION"
echo "Report:"
echo "  $REPORT"
if [[ -n "$ZIP" ]]; then
  echo "Zip:"
  echo "  $ZIP"
fi
