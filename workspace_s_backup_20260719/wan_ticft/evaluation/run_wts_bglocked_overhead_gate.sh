#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/manifests/val_ticft_balanced_125_frames.jsonl}
LIMIT_PER_DOMAIN=${LIMIT_PER_DOMAIN:-24}
OUT=${OUT:-$ROOT/wts_bglocked_overhead${LIMIT_PER_DOMAIN}}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== WTS overhead/fixed background-locked actor composite gate ==="
echo "out=$OUT"
echo "limit_per_domain=$LIMIT_PER_DOMAIN"

rm -rf "$OUT"

echo
echo "[1/3] Materialize candidates"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --limit_per_domain "$LIMIT_PER_DOMAIN" \
  --views wts_overhead_or_fixed_camera \
  --frames 50 \
  --cover_manifest_targets \
  --candidates \
    region_norm_long24 \
    actor_layer_small_filled \
    bg_locked_region_long24 \
    bg_locked_actor_small \
  --clean

echo
echo "[2/3] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_report_pixels \
  --target_report "$OUT/target_report.csv" \
  --report "region_long24=$OUT/region_norm_long24_report.csv" \
  --report "actor_small=$OUT/actor_layer_small_filled_report.csv" \
  --report "bg_locked_long24=$OUT/bg_locked_region_long24_report.csv" \
  --report "bg_locked_actor_small=$OUT/bg_locked_actor_small_report.csv" \
  --out_csv "$ROOT/wts_bglocked_overhead${LIMIT_PER_DOMAIN}_pixel.csv"

echo
echo "[3/3] FID/FVD proxy"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --target_report "$OUT/target_report.csv" \
  --full_video \
  --report "region_long24=$OUT/region_norm_long24_report.csv" \
  --report "actor_small=$OUT/actor_layer_small_filled_report.csv" \
  --report "bg_locked_long24=$OUT/bg_locked_region_long24_report.csv" \
  --report "bg_locked_actor_small=$OUT/bg_locked_actor_small_report.csv" \
  --out_csv "$ROOT/wts_bglocked_overhead${LIMIT_PER_DOMAIN}_proxy.csv"

echo
echo "Gate rule:"
echo "  - Continue only if a bg_locked candidate improves FID/FVD versus actor_small"
echo "    without a material PSNR/AbsDiff collapse."
echo "  - If bg_locked loses to actor_small, close this cheap composite route before"
echo "    trying SA2VA/SAM2, because the mask source is probably not the bottleneck."
echo
echo "pixel=$ROOT/wts_bglocked_overhead${LIMIT_PER_DOMAIN}_pixel.csv"
echo "proxy=$ROOT/wts_bglocked_overhead${LIMIT_PER_DOMAIN}_proxy.csv"
