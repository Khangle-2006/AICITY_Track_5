#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
TEST_CSV=${TEST_CSV:-$INDEX/test_index.csv}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
BASE_BDD=${BASE_BDD:-$ROOT/cnn_ltxmix65_bdd_test}
CHECKPOINT=${CHECKPOINT:-$ROOT/temporal_latent_feature_gate24/temporal_latent_refiner.pt}
BDD_OUT=${BDD_OUT:-$ROOT/temporal_latent_feature_bdd_test}
WTS=${WTS:-$ROOT/wts_overhead_object_layer_vehicle_medium_test}
SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate}
REPORT=${REPORT:-$ROOT/hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate_report.csv}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Build temporal latent feature BDD + WTS overhead-object candidate ==="
echo "BASE_BDD=$BASE_BDD"
echo "CHECKPOINT=$CHECKPOINT"
echo "BDD_OUT=$BDD_OUT"
echo "WTS=$WTS"
echo "SUBMISSION=$SUBMISSION"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -d "$BASE_BDD" ]]; then
  echo "Missing base BDD test directory: $BASE_BDD" >&2
  exit 2
fi

echo
echo "[1/4] Apply temporal latent feature refiner to BDD test"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
"$PYTHON" -m wan_ticft.evaluation.materialize_temporal_latent_refiner_test \
  --test_csv "$TEST_CSV" \
  --base_dir "$BASE_BDD" \
  --checkpoint "$CHECKPOINT" \
  --out_dir "$BDD_OUT" \
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
  echo "[2/4] Reuse existing WTS: $WTS"
fi

echo
echo "[3/4] Build hybrid submission directory, no zip"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_OUT" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label temporal_latent_feature \
  --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo
echo "[4/4] Verify full submission folder"
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
