#!/usr/bin/env bash
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$SOLUTION_DIR:${PYTHONPATH:-}"

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
TEST_CSV=${TEST_CSV:-$INDEX/test_index.csv}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}

# This folder is the BDD-only generative base for the tracked public 69.3815
# route. Reusing the exact same base frames is required for byte-identical
# reproduction of the submitted candidate.
LTXHF_BASE=${LTXHF_BASE:-$ROOT/ltxhf_t55_bdd_test}

BDD_RENDERED=${BDD_RENDERED:-$ROOT/depth_multiframe_mf_h8_ltxhf_t55_bdd_test}
MODEL_CACHE=${MODEL_CACHE:-$ROOT/depth_layered_models}
WTS=${WTS:-$ROOT/wts_overhead_object_layer_vehicle_medium_test}
SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate}
REPORT=${REPORT:-$ROOT/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate_report.csv}
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Build multiframe mf_h8_ltxhf_t55 + WTS candidate ==="
echo "solution_dir=$SOLUTION_DIR"
echo "index=$INDEX"
echo "ltxhf_base=$LTXHF_BASE"
echo "bdd_rendered=$BDD_RENDERED"
echo "wts=$WTS"
echo "submission=$SUBMISSION"
echo "report=$REPORT"

for path in "$TEST_CSV" "$LTXHF_BASE/report.csv"; do
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

echo
echo "[1/4] Render BDD test with multiframe depth-layered history"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME="$MODEL_CACHE/hf_home" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_multiframe_test \
  --test_csv "$TEST_CSV" \
  --base_dir "$LTXHF_BASE" \
  --out_dir "$BDD_RENDERED" \
  --model_cache "$MODEL_CACHE/huggingface" \
  --view_type bdd \
  --alpha 0.75 \
  --max_history 8 \
  --history_stride 2 \
  --base_mode video \
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

if [[ ! -d "$WTS" ]]; then
  echo
  echo "[2/4] Generate WTS overhead object-layer + vehicle medium"
  "$PYTHON" -m baseline.cli \
    --index_dir "$INDEX" \
    --out "$WTS" \
    --view_type wts_overhead_or_fixed_camera \
    --method actor_layer_small_filled \
    --actor_layer_decay 0.985 \
    --actor_layer_max_flow_scale 12.0 \
    --actor_layer_max_area_frac 0.025

  "$PYTHON" -m baseline.cli \
    --index_dir "$INDEX" \
    --out "$WTS" \
    --view_type wts_vehicle_or_ip_camera \
    --method region_norm_medium \
    --region_norm_decay 0.96 \
    --region_norm_max_flow_scale 6.0
else
  echo
  echo "[2/4] Reuse existing WTS outputs: $WTS"
fi

echo
echo "[3/4] Build hybrid submission directory"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_RENDERED" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label multiframe_mf_h8_ltxhf_t55 \
  --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo
echo "[4/4] Verify complete submission folder"
"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$SUBMISSION" \
  --submission_dir "$SUBMISSION" \
  --verify_only

echo
echo "Candidate verified but not zipped:"
echo "  $SUBMISSION"
echo "Report:"
echo "  $REPORT"
