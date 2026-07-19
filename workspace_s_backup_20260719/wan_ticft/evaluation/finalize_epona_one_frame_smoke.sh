#!/usr/bin/env bash
set -euo pipefail

OUT=/workspace/experiments/wan_ticft/epona_one_frame_smoke
REPO=/workspace/third_party/Epona
SAMPLE=val_bdd_pc_5k_video2149_phase0
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

/opt/conda/envs/wan22/bin/python /workspace/wan_ticft/evaluation/evaluate_epona_one_frame.py \
  --prediction "$REPO/test_videos/one_frame/sliding_0/10.png" \
  --baseline "$OUT/data/$SAMPLE/000009.png" \
  --reference "$OUT/reference/next_5fps.png" \
  --out_json "$OUT/metrics.json"

nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader -i "$GPU_ID" | tee "$OUT/gpu_after.txt"

echo "runtime:"
cat "$OUT/runtime.txt"
