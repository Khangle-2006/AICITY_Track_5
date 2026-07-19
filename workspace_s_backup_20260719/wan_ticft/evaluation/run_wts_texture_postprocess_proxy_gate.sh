#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
SOURCE=${SOURCE:-$ROOT/wts_actor_guard_gate20}
ROUTED=${ROUTED:-$SOURCE/routed_reports/current_overhead_long24_vehicle_medium_report.csv}
TARGET=${TARGET:-$SOURCE/target_report.csv}
OUT=${OUT:-$ROOT/wts_texture_postprocess_gate20}
MANIFEST=${MANIFEST:-$ROOT/manifests/val_ticft_balanced_125_frames.jsonl}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== WTS texture postprocess FID/FVD proxy gate ==="
echo "Source route: ${ROUTED}"
echo "Target report: ${TARGET}"

if [[ ! -f "$ROUTED" || ! -f "$TARGET" ]]; then
  echo "Missing routed WTS validation reports. Run actor-guard routed proxy setup first:" >&2
  echo "  bash /workspace/wan_ticft/evaluation/run_wts_actor_guard_distribution_gate.sh" >&2
  echo "  bash /workspace/wan_ticft/evaluation/run_wts_actor_guard_routed_proxy_gate.sh" >&2
  exit 2
fi

rm -rf "$OUT"
mkdir -p "$OUT"

materialize() {
  local label="$1"
  local mode="$2"
  local sigma="$3"
  local amount="$4"
  local clip="$5"
  "$PYTHON" -m wan_ticft.evaluation.materialize_video_postprocess_report \
    --report "$ROUTED" \
    --out_dir "$OUT/$label" \
    --label "$label" \
    --mode "$mode" \
    --sigma "$sigma" \
    --amount "$amount" \
    --clahe_clip "$clip" \
    --clean
}

echo
echo "[1/3] Materialize lightweight texture candidates"
materialize unsharp25 unsharp 1.0 0.25 1.2
materialize unsharp40 unsharp 1.0 0.40 1.2
materialize clahe12 clahe 1.0 0.25 1.2
materialize unsharp25_clahe12 unsharp_clahe 1.0 0.25 1.2

echo
echo "[2/3] Relative FID/FVD proxy"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --target_report "$TARGET" \
  --full_video \
  --report "current=$ROUTED" \
  --report "unsharp25=$OUT/unsharp25/report.csv" \
  --report "unsharp40=$OUT/unsharp40/report.csv" \
  --report "clahe12=$OUT/clahe12/report.csv" \
  --report "unsharp25_clahe12=$OUT/unsharp25_clahe12/report.csv" \
  --out_csv "$ROOT/wts_texture_postprocess_proxy_gate20.csv"

echo
echo "[3/3] Contact sheets for visual QA"
"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/unsharp25/videos" \
  --mode videos \
  --out_dir "$OUT/unsharp25_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4
"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/unsharp40/videos" \
  --mode videos \
  --out_dir "$OUT/unsharp40_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

echo
echo "Gate rule:"
echo "  - Promote only if a postprocess improves both FID and FVD proxies versus current."
echo "  - Reject if visual QA shows ringing, edge halos, flicker, or over-contrast."
echo "  - Pixel metrics are not required for this gate; this is distribution-only."
