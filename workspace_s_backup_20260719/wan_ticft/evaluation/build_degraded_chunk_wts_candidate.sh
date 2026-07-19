#!/bin/bash
set -euo pipefail
# Degrade chunk-base depth-warp BDD output toward real-camera statistics (blur+noise+JPEG),
# then merge with WTS. Tests whether realism degradation cuts FID enough for chunk to beat a75.
cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=/workspace/experiments/index
ROOT=/workspace/experiments/wan_ticft
BASE=${BASE:-$ROOT/depth_chunkbase_fixed_dpt_bdd_test}   # chunk + depth-warp BDD (the 66.28 BDD half)
WTS=${WTS:-$ROOT/wts_overhead_object_layer_vehicle_medium_test}
BLUR=${BLUR:-0.6}; NOISE=${NOISE:-4.0}; JPEG_Q=${JPEG_Q:-35}; DOWNSCALE=${DOWNSCALE:-1.0}
TAG=${TAG:-b06n4j35}
DEGRADED=$ROOT/depth_chunkbase_degr_${TAG}_bdd_test
SUBMISSION=/workspace/experiments/hybrid_chunk_degr_${TAG}_wts_candidate
REPORT=$ROOT/hybrid_chunk_degr_${TAG}_wts_candidate_report.csv

echo "=== Degraded chunk candidate (blur=$BLUR noise=$NOISE jpeg_q=$JPEG_Q downscale=$DOWNSCALE) ==="
for p in "$INDEX/test_index.csv" "$BASE" "$WTS"; do [[ -e "$p" ]] || { echo "Missing: $p" >&2; exit 2; }; done

echo "[1/3] Degrade chunk BDD frames toward real-camera statistics"
"$PYTHON" -m wan_ticft.evaluation.degrade_frames \
  --base_dir "$BASE" --out_dir "$DEGRADED" \
  --blur "$BLUR" --noise "$NOISE" --jpeg_q "$JPEG_Q" --downscale "$DOWNSCALE" --clean

echo "[2/3] Merge with WTS object-layer"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" --wan_dir "$DEGRADED" --region_dir "$WTS" \
  --out_dir "$SUBMISSION" --report_csv "$REPORT" \
  --wan_label chunk_degr_${TAG} --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera --clean

echo "[3/3] Verify"
"$PYTHON" -m baseline.submission --index_dir "$INDEX" --gen_dir "$SUBMISSION" --submission_dir "$SUBMISSION" --verify_only
echo; echo "Submission ready: $SUBMISSION"; echo "Report: $REPORT"
