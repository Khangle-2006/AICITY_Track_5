#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
OUT="$ROOT/wts_actor_guard_gate20"
MANIFEST="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

for VIEW in wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera; do
  SHORT="${VIEW#wts_}"
  echo "=== WTS actor-guard proxy by domain: $VIEW ==="
  /workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  TORCH_HOME="$ROOT/tools/torch_cache" \
  "$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
    --manifest "$MANIFEST" \
    --view_type "$VIEW" \
    --target_report "$OUT/target_report.csv" \
    --full_video \
    --report "region_medium=$OUT/region_norm_medium_report.csv" \
    --report "region_long24=$OUT/region_norm_long24_report.csv" \
    --report "guard25=$OUT/region_actor_guard25_report.csv" \
    --report "guard50=$OUT/region_actor_guard50_report.csv" \
    --report "long24_guard50=$OUT/region_long24_actor_guard50_report.csv" \
    --out_csv "$ROOT/wts_actor_guard_${SHORT}_proxy_gate10.csv"

  QA="$ROOT/wts_actor_guard_${SHORT}_visual_qa"
  "$PYTHON" -m wan_ticft.evaluation.prepare_report_video_subset \
    --view_type "$VIEW" \
    --report "target=$OUT/target_report.csv" \
    --report "region_medium=$OUT/region_norm_medium_report.csv" \
    --report "region_long24=$OUT/region_norm_long24_report.csv" \
    --report "guard25=$OUT/region_actor_guard25_report.csv" \
    --report "guard50=$OUT/region_actor_guard50_report.csv" \
    --report "long24_guard50=$OUT/region_long24_actor_guard50_report.csv" \
    --out_dir "$QA" \
    --clean

  for LABEL in target region_medium region_long24 guard25 guard50 long24_guard50; do
    "$PYTHON" -m baseline.video_contact_sheet \
      --input "$QA/$LABEL" \
      --mode videos \
      --out_dir "${QA}_${LABEL}_sheets" \
      --samples 16 \
      --tile_width 320 \
      --cols 4
  done
done

echo
echo "Visual QA rule:"
echo "  - Continue only if a guard route reduces smear without freezing"
echo "    pedestrians/vehicles or creating visible mask halos."
echo "  - Inspect each domain's target/medium/long24/guard25/guard50/"
echo "    long24_guard50 sheets."
