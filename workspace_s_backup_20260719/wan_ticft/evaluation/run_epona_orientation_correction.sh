#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft
REPO=/workspace/third_party/Epona
OUT="$ROOT/epona_orientation_correction"
MANIFEST="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
SAMPLE=val_bdd_pc_5k_video2149_phase0
MODEL="$ROOT/epona_pretrained/epona_nuplan+nusc.pkl"
PYTHON="$ROOT/epona_env/bin/python"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Epona orientation-aware one-frame correction ==="
bash /workspace/wan_ticft/evaluation/setup_epona_smoke.sh
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 30000

rm -rf "$OUT"
rm -rf "$REPO/test_videos/orientation_correction"
mkdir -p "$OUT"
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.prepare_epona_demo \
  --manifest "$MANIFEST" \
  --sample_id "$SAMPLE" \
  --out_dir "$OUT" \
  --orientation portrait_ccw

cd "$REPO"
START_SECONDS="$(date +%s)"
"$PYTHON" scripts/test/test_demo.py \
  --exp_name orientation_correction \
  --start_id 0 \
  --end_id 1 \
  --data_root "$OUT/data" \
  --resume_path "$MODEL" \
  --config /workspace/wan_ticft/evaluation/epona_one_frame_config.py \
  2>&1 | tee "$OUT/run.log"
END_SECONDS="$(date +%s)"
echo "elapsed_seconds=$((END_SECONDS - START_SECONDS))" | tee "$OUT/runtime.txt"

/opt/conda/envs/wan22/bin/python /workspace/wan_ticft/evaluation/evaluate_epona_one_frame.py \
  --prediction "$REPO/test_videos/orientation_correction/sliding_0/10.png" \
  --prediction_rotation clockwise \
  --baseline "$ROOT/epona_one_frame_smoke/data/$SAMPLE/000009.png" \
  --reference "$OUT/reference/next_5fps.png" \
  --out_json "$OUT/metrics.json"

nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader -i "$GPU_ID" | tee "$OUT/gpu_after.txt"

echo
echo "Correction rule:"
echo "  - Continue only if orientation-aware Epona still beats repeat-last and"
echo "    improves over native Epona's 16.9874 PSNR / 0.4972 SSIM."
echo "  - If it fails, retain native orientation only for further feasibility."
