#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
TEST_CSV=${TEST_CSV:-$INDEX/test_index.csv}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
BDD_BASE=${BDD_BASE:-$ROOT/cnn_ltxmix65_bdd_test}
BDD_DEPTH=${BDD_DEPTH:-$ROOT/depth_layered_a75_bdd_test}
BDD_REPEAT=${BDD_REPEAT:-$ROOT/repeat_last_bdd_test_for_phase12}
BDD_PHASE=${BDD_PHASE:-$ROOT/bdd_phase12_repeat_depth_test}
MODEL_CACHE=${MODEL_CACHE:-$ROOT/depth_layered_models}
WTS=${WTS:-$ROOT/wts_overhead_object_layer_vehicle_medium_test}
SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_bdd_phase12_repeat_depth_wts_overhead_object_layer_candidate}
REPORT=${REPORT:-$ROOT/hybrid_bdd_phase12_repeat_depth_wts_overhead_object_layer_candidate_report.csv}
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Build BDD phase12 repeat/depth + current WTS overhead-object candidate ==="
echo "No zip is created."
echo "BDD_PHASE=$BDD_PHASE"
echo "WTS=$WTS"
echo "SUBMISSION=$SUBMISSION"

for path in "$TEST_CSV" "$BDD_BASE/report.csv"; do
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
[[ -d "$WTS" ]] || { echo "Missing WTS source: $WTS" >&2; exit 2; }

echo
echo "[1/5] Render/reuse BDD depth-layered a75"
if [[ -s "$BDD_DEPTH/report.csv" ]]; then
  echo "  found existing $BDD_DEPTH/report.csv; skip render"
else
  /workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 12000
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  HF_HOME="$MODEL_CACHE/hf_home" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer_test \
    --test_csv "$TEST_CSV" \
    --base_dir "$BDD_BASE" \
    --out_dir "$BDD_DEPTH" \
    --model_cache "$MODEL_CACHE/huggingface" \
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
fi

echo
echo "[2/5] Materialize BDD repeat-last aligned to depth output"
"$PYTHON" -m wan_ticft.evaluation.materialize_repeat_last_test \
  --test_csv "$TEST_CSV" \
  --reference_dir "$BDD_DEPTH" \
  --out_dir "$BDD_REPEAT" \
  --view_type bdd \
  --clean

echo
echo "[3/5] Route BDD phase1/phase2 to repeat, all other BDD to depth"
"$PYTHON" -m wan_ticft.evaluation.materialize_phase_rule_test \
  --test_csv "$TEST_CSV" \
  --out_dir "$BDD_PHASE" \
  --default_dir "$BDD_DEPTH" \
  --default_label depth_a75 \
  --phase_dir "1=$BDD_REPEAT=repeat_last" \
  --phase_dir "2=$BDD_REPEAT=repeat_last" \
  --view_type bdd \
  --clean

echo
echo "[4/5] Build hybrid submission directory with unchanged WTS"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_PHASE" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label bdd_phase12_repeat_depth \
  --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo
echo "[5/5] Verify full submission folder"
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
echo "BDD phase route report:"
echo "  $BDD_PHASE/report.csv"
