#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
OUT="$ROOT/wts_actor_guard_gate20"
MANIFEST="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== WTS actor/vehicle anti-smear gate: 10 samples/domain ==="
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --limit_per_domain 10 \
  --frames 50 \
  --candidates \
    repeat_last \
    region_norm_medium \
    region_norm_long24 \
    region_actor_guard25 \
    region_actor_guard50 \
    region_long24_actor_guard50 \
  --clean

echo
echo "=== WTS anti-smear relative FID/FVD proxy ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --target_report "$OUT/target_report.csv" \
  --full_video \
  --report "repeat_last=$OUT/repeat_last_report.csv" \
  --report "region_medium=$OUT/region_norm_medium_report.csv" \
  --report "region_long24=$OUT/region_norm_long24_report.csv" \
  --report "guard25=$OUT/region_actor_guard25_report.csv" \
  --report "guard50=$OUT/region_actor_guard50_report.csv" \
  --report "long24_guard50=$OUT/region_long24_actor_guard50_report.csv" \
  --out_csv "$ROOT/wts_actor_guard_distribution_proxy_gate20.csv"

echo
echo "Gate rule:"
echo "  - Continue only if an actor-guard variant improves both FID and FVD"
echo "    proxies versus the current routed baseline components."
echo "  - It must not visibly freeze pedestrians/vehicles or create mask halos."
echo "  - If it fails, close hand-engineered WTS anti-smear compositing."
