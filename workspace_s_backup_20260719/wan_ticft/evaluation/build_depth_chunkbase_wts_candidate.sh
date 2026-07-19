#!/bin/bash
set -euo pipefail
# Candidate: depth-layered warp ON TOP OF ltx2b CHUNK base (1280x720), + adaptive blend, merged with WTS.
# Goal: keep chunk sharpness but pull FID down via real-pixel reprojection. Directly submittable.
cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=/workspace/experiments/index
TEST_CSV="$INDEX/test_index.csv"
ROOT=/workspace/experiments/wan_ticft
BASE=${BASE:-$ROOT/ltx2b_chunked_1280x720_bdd_test}          # chunk test as base
MODEL_CACHE=${MODEL_CACHE:-$ROOT/depth_layered_models}
WTS=${WTS:-$ROOT/wts_overhead_object_layer_vehicle_medium_test}
ALPHA=${ALPHA:-0.75}
ADAPTIVE=${ADAPTIVE:-1}                  # 1 = --adaptive_blend
TEMPORAL_DECAY=${TEMPORAL_DECAY:-0}      # 1 = --temporal_decay
DEPTH_BACKEND=${DEPTH_BACKEND:-dpt}      # dpt | da3_mono | da3_metric
TAG=${TAG:-adaptive}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"; GPU_ID="${GPU_ID%%,*}"

BDD_RENDERED=$ROOT/depth_chunkbase_${TAG}_${DEPTH_BACKEND}_bdd_test
SUBMISSION=/workspace/experiments/hybrid_depth_chunkbase_${TAG}_${DEPTH_BACKEND}_wts_candidate
REPORT=$ROOT/hybrid_depth_chunkbase_${TAG}_${DEPTH_BACKEND}_wts_candidate_report.csv

echo "=== Candidate: chunk-base depth-warp (tag=$TAG backend=$DEPTH_BACKEND alpha=$ALPHA adaptive=$ADAPTIVE tdecay=$TEMPORAL_DECAY) ==="
echo "BASE=$BASE"; echo "BDD_RENDERED=$BDD_RENDERED"; echo "SUBMISSION=$SUBMISSION"
for p in "$TEST_CSV" "$BASE/report.csv" "$WTS"; do [[ -e "$p" ]] || { echo "Missing: $p" >&2; exit 2; }; done
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 12000 || true

EXTRA=()
[[ "$ADAPTIVE" == "1" ]] && EXTRA+=( --adaptive_blend )
[[ "$TEMPORAL_DECAY" == "1" ]] && EXTRA+=( --temporal_decay )

echo "[1/3] Render depth-warp on chunk base (BDD test, 44 samples)"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME="$MODEL_CACHE/hf_home" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer_test \
  --test_csv "$TEST_CSV" \
  --base_dir "$BASE" \
  --out_dir "$BDD_RENDERED" \
  --model_cache "$MODEL_CACHE/huggingface" \
  --depth_backend "$DEPTH_BACKEND" \
  --view_type bdd \
  --alpha "$ALPHA" \
  "${EXTRA[@]}" \
  --focal_scale 0.9 --point_stride 2 --splat_radius 1 \
  --min_matches 20 --min_inliers 12 --photo_sigma 32 \
  --segment_batch 4 --pose_smoothing_window 7 \
  --filled_pose_scale 0.35 --confidence_ema 0.65 \
  --clean

echo "[2/3] Merge with WTS object-layer -> 71-folder submission"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_RENDERED" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label depth_chunkbase_${TAG}_${DEPTH_BACKEND} \
  --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo "[3/3] Verify"
"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" --gen_dir "$SUBMISSION" --submission_dir "$SUBMISSION" --verify_only

echo
echo "Submission ready: $SUBMISSION"
echo "Report: $REPORT"
