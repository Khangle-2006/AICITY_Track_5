#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft
REPO=/workspace/third_party/Epona
OUT="$ROOT/epona_one_frame_gate4"
MANIFEST="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
MODEL="$ROOT/epona_pretrained/epona_nuplan+nusc.pkl"
PYTHON="$ROOT/epona_env/bin/python"
SAMPLES=(
  val_bdd_pc_5k_video7040_phase0
  val_bdd_pc_5k_video1407_phase0
  val_bdd_pc_5k_video3237_phase0
  val_bdd_pc_5k_video5447_phase3
)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Epona native one-frame generalization gate: 4 new BDD samples ==="
bash /workspace/wan_ticft/evaluation/setup_epona_smoke.sh
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 30000

rm -rf "$OUT"
rm -rf "$REPO/test_videos/gate4"
mkdir -p "$OUT/data" "$OUT/reference" "$OUT/prep"
for sample in "${SAMPLES[@]}"; do
  prep="$OUT/prep/$sample"
  /opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.prepare_epona_demo \
    --manifest "$MANIFEST" \
    --sample_id "$sample" \
    --out_dir "$prep"
  mv "$prep/data/$sample" "$OUT/data/$sample"
  cp "$prep/reference/next_5fps.png" "$OUT/reference/$sample.png"
done

cd "$REPO"
START_SECONDS="$(date +%s)"
"$PYTHON" scripts/test/test_demo.py \
  --exp_name gate4 \
  --start_id 0 \
  --end_id 4 \
  --data_root "$OUT/data" \
  --resume_path "$MODEL" \
  --config /workspace/wan_ticft/evaluation/epona_one_frame_config.py \
  2>&1 | tee "$OUT/run.log"
END_SECONDS="$(date +%s)"
echo "elapsed_seconds=$((END_SECONDS - START_SECONDS))" | tee "$OUT/runtime.txt"

/opt/conda/envs/wan22/bin/python /workspace/wan_ticft/evaluation/evaluate_epona_one_frame_gate.py \
  --data_root "$OUT/data" \
  --reference_root "$OUT/reference" \
  --prediction_root "$REPO/test_videos/gate4" \
  --out_csv "$OUT/metrics.csv"

echo
echo "Gate rule:"
echo "  - Pass only if macro PSNR improves by >=0.30, macro SSIM does not fall,"
echo "    at least 3/4 samples win, and no sample loses more than 0.50 PSNR."
echo "  - If pass, test a short multi-frame rollout before any larger gate."
echo "  - If fail, close Epona without checkpoint or pose tuning."
