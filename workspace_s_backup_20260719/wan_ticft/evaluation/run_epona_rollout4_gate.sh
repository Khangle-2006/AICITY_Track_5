#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft
REPO=/workspace/third_party/Epona
OUT="$ROOT/epona_rollout4_gate"
MANIFEST="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
MODEL="$ROOT/epona_pretrained/epona_nuplan+nusc.pkl"
PYTHON="$ROOT/epona_env/bin/python"
SAMPLES=(
  val_bdd_pc_5k_video3237_phase0
  val_bdd_pc_5k_video5447_phase3
)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Epona four-frame autoregressive rollout gate: strong + night failure ==="
bash /workspace/wan_ticft/evaluation/setup_epona_smoke.sh
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 30000

rm -rf "$OUT"
rm -rf "$REPO/test_videos/rollout4"
mkdir -p "$OUT/data" "$OUT/reference" "$OUT/prep"
for sample in "${SAMPLES[@]}"; do
  prep="$OUT/prep/$sample"
  /opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.prepare_epona_demo \
    --manifest "$MANIFEST" \
    --sample_id "$sample" \
    --out_dir "$prep" \
    --future_frames 4
  mv "$prep/data/$sample" "$OUT/data/$sample"
  mv "$prep/reference" "$OUT/reference/$sample"
done

cd "$REPO"
START_SECONDS="$(date +%s)"
"$PYTHON" scripts/test/test_demo.py \
  --exp_name rollout4 \
  --start_id 0 \
  --end_id 2 \
  --data_root "$OUT/data" \
  --resume_path "$MODEL" \
  --config /workspace/wan_ticft/evaluation/epona_rollout4_config.py \
  2>&1 | tee "$OUT/run.log"
END_SECONDS="$(date +%s)"
echo "elapsed_seconds=$((END_SECONDS - START_SECONDS))" | tee "$OUT/runtime.txt"

/opt/conda/envs/wan22/bin/python /workspace/wan_ticft/evaluation/evaluate_epona_rollout_gate.py \
  --data_root "$OUT/data" \
  --reference_root "$OUT/reference" \
  --prediction_root "$REPO/test_videos/rollout4" \
  --future_frames 4 \
  --out_csv "$OUT/metrics.csv"

echo
echo "Rollout gate rule:"
echo "  - Pass only if aggregate PSNR and SSIM beat repeat-last,"
echo "    step-4 PSNR does not fall below repeat-last, and worst loss is > -0.75."
echo "  - If pass, expand to a balanced rollout gate."
echo "  - If fail, close Epona rollout; one-frame routing alone is not deployable."
