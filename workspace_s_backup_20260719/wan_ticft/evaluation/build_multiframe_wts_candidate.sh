#!/bin/bash
set -euo pipefail
cd /workspace
export CPU_THREADS="${CPU_THREADS:-4}"
export OMP_NUM_THREADS=$CPU_THREADS OPENBLAS_NUM_THREADS=$CPU_THREADS MKL_NUM_THREADS=$CPU_THREADS NUMEXPR_NUM_THREADS=$CPU_THREADS VECLIB_MAXIMUM_THREADS=$CPU_THREADS
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=/workspace/experiments/index
TEST_CSV="$INDEX/test_index.csv"
ROOT=/workspace/experiments/wan_ticft
BASE=${BASE:-$ROOT/cnn_ltxmix65_bdd_test}
MODEL_CACHE=${MODEL_CACHE:-$ROOT/depth_layered_models}
WTS=${WTS:-$ROOT/wts_overhead_object_layer_vehicle_medium_test}
MAXH="${MAX_HISTORY:-8}"; HSTRIDE="${HISTORY_STRIDE:-2}"; CHUNK_DIR="${CHUNK_DIR:-}"; BASE_MODE="${BASE_MODE:-video}"
TAG="${TAG:-mf_h${MAXH}}"
BDD_OUT=$ROOT/depth_multiframe_${TAG}_bdd_test
SUBMISSION=/workspace/experiments/hybrid_multiframe_${TAG}_wts_candidate
REPORT=$ROOT/hybrid_multiframe_${TAG}_wts_candidate_report.csv
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"; GPU_ID="${GPU_ID%%,*}"

echo "=== Multi-frame fusion candidate (max_history=$MAXH stride=$HSTRIDE) ==="
for p in "$TEST_CSV" "$BASE/report.csv" "$WTS"; do [[ -e "$p" ]] || { echo "Missing $p" >&2; exit 2; }; done

echo "[1/3] Render multi-frame fusion on BDD test"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME="$MODEL_CACHE/hf_home" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_multiframe_test \
  --test_csv "$TEST_CSV" --base_dir "$BASE" --out_dir "$BDD_OUT" --model_cache "$MODEL_CACHE/huggingface" \
  --view_type bdd --alpha 0.75 --max_history "$MAXH" --history_stride "$HSTRIDE" --base_mode "$BASE_MODE" \
  --focal_scale 0.9 --point_stride 2 --splat_radius 1 --min_matches 20 --min_inliers 12 \
  --photo_sigma 32 --segment_batch 4 --pose_smoothing_window 7 --filled_pose_scale 0.35 --confidence_ema 0.65 ${CHUNK_DIR:+--chunk_dir "$CHUNK_DIR"} --clean

echo "[2/3] Merge with WTS"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission --index_dir "$INDEX" --wan_dir "$BDD_OUT" --region_dir "$WTS" \
  --out_dir "$SUBMISSION" --report_csv "$REPORT" --wan_label multiframe_${TAG} \
  --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera --clean

echo "[3/3] Verify"
"$PYTHON" -m baseline.submission --index_dir "$INDEX" --gen_dir "$SUBMISSION" --submission_dir "$SUBMISSION" --verify_only
echo; echo "Submission ready: $SUBMISSION"
