#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft
REPO=/workspace/third_party/Epona
OUT="$ROOT/epona_one_frame_smoke"
MANIFEST="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
SAMPLE=val_bdd_pc_5k_video2149_phase0
MODEL="$ROOT/epona_pretrained/epona_nuplan+nusc.pkl"
VAE="$ROOT/epona_pretrained/dcae_td_20000.pkl"
PYTHON="$ROOT/epona_env/bin/python"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Epona one-frame custom-input feasibility smoke ==="
bash /workspace/wan_ticft/evaluation/setup_epona_smoke.sh
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 30000

rm -rf "$OUT"
rm -rf "$REPO/test_videos/one_frame"
mkdir -p "$OUT"
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.prepare_epona_demo \
  --manifest "$MANIFEST" \
  --sample_id "$SAMPLE" \
  --out_dir "$OUT"

cd "$REPO"
START_SECONDS="$(date +%s)"
"$PYTHON" scripts/test/test_demo.py \
  --exp_name one_frame \
  --start_id 0 \
  --end_id 1 \
  --data_root "$OUT/data" \
  --resume_path "$MODEL" \
  --config /workspace/wan_ticft/evaluation/epona_one_frame_config.py \
  2>&1 | tee "$OUT/run.log"
END_SECONDS="$(date +%s)"
ELAPSED_SECONDS="$((END_SECONDS - START_SECONDS))"
echo "elapsed_seconds=$ELAPSED_SECONDS" | tee "$OUT/runtime.txt"

/opt/conda/envs/wan22/bin/python /workspace/wan_ticft/evaluation/evaluate_epona_one_frame.py \
  --prediction "$REPO/test_videos/one_frame/sliding_0/10.png" \
  --baseline "$OUT/data/$SAMPLE/000009.png" \
  --reference "$OUT/reference/next_5fps.png" \
  --out_json "$OUT/metrics.json"

nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader -i "$GPU_ID" | tee "$OUT/gpu_after.txt"

echo
echo "Gate rule:"
echo "  - Pass feasibility only if inference completes on one GPU, output is finite,"
echo "    beats repeat-last, and elapsed time implies a practical required horizon."
echo "  - Review the generated frame for identity and geometry before any larger gate."
echo "  - If runtime is impractical or the first frame is structurally poor, close Epona."
