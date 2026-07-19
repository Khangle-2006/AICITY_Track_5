#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/vace_region_bdd_gate_video1170

echo "=== VACE region-control gate: one BDD sample, one configuration ==="
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 30000

export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=3600
PYTHONPATH=/workspace:/workspace/third_party/DiffSynth-Studio \
/opt/conda/envs/wan22/bin/python -m wan_ticft.inference.run_vace_region_smoke \
  --index_dir /workspace/experiments/index \
  --region_dir /workspace/experiments/region_norm_medium_test_smoke_bdd \
  --out_dir "$OUT" \
  --sample video1170 \
  --width 832 \
  --height 480 \
  --num_frames 49 \
  --steps 10 \
  --seed 1

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$OUT/videos" \
  --mode videos \
  --out_dir "${OUT}_sheets" \
  --samples 12 \
  --tile_width 320 \
  --cols 4

echo "Inspect: ${OUT}_sheets"
