#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
INDEX=/workspace/experiments/index
ROOT=/workspace/experiments/wan_ticft
TEST_CSV="$INDEX/test_index.csv"
MANIFEST="$ROOT/manifests/test_ticft_frames.jsonl"
BDD_BASE="$ROOT/cnn_ltxmix65_bdd_test"
BDD_DEPTH="$ROOT/depth_layered_a75_bdd_test"
BDD_OUT="$ROOT/sa2va_actor_depth_a100_bdd_test"
WTS="$ROOT/wts_overhead_object_layer_vehicle_medium_test"
SUBMISSION=/workspace/experiments/hybrid_sa2va_actor_depth_a100_wts_overhead_object_candidate
REPORT="$ROOT/hybrid_sa2va_actor_depth_a100_wts_overhead_object_candidate_report.csv"
MODEL="${SA2VA_MODEL:-ByteDance/Sa2VA-4B}"
REVISION="${SA2VA_REVISION:-3fee777d49ee9276eac51ea3e5f9b69e81d09be6}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
CHUNK_FRAMES="${SA2VA_CHUNK_FRAMES:-10}"
CLEAN="${CLEAN:-0}"

echo "=== Build Sa2VA actor-protected depth a100 candidate ==="
echo "GPU=$GPU_ID chunk_frames=$CHUNK_FRAMES clean=$CLEAN"
echo "Per-sample resume is enabled. No zip is created."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 32000
for path in "$TEST_CSV" "$MANIFEST" "$BDD_BASE/report.csv" "$BDD_DEPTH/report.csv"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
[[ -d "$WTS" ]] || { echo "Missing WTS source: $WTS" >&2; exit 2; }

CLEAN_ARG=()
if [[ "$CLEAN" == "1" ]]; then
  CLEAN_ARG+=(--clean)
fi

echo "[1/3] Materialize actor-aware BDD test frames"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
"$PYTHON" -m wan_ticft.evaluation.materialize_sa2va_actor_aware_depth_test \
  --test_csv "$TEST_CSV" \
  --manifest "$MANIFEST" \
  --base_dir "$BDD_BASE" \
  --depth_dir "$BDD_DEPTH" \
  --out_dir "$BDD_OUT" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --chunk_frames "$CHUNK_FRAMES" \
  "${CLEAN_ARG[@]}"

echo "[2/3] Build hybrid with unchanged WTS"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_OUT" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label sa2va_actor_depth_a100 \
  --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo "[3/3] Verify complete 71-sample folder"
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
echo "BDD diagnostics:"
echo "  $BDD_OUT/diagnostics.csv"
