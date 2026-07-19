#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/ltxv_bdd_behavior_prompt_correction4
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
DENSE_MANIFEST="$OUT/dense40_manifest.jsonl"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 24000

echo "=== LTX BDD behavior-only prompt correction: same four samples ==="
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.run_ltxv_val_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --ltx_repo /workspace/third_party/LTX-Video \
  --ltx_python "$ROOT/ltxv_env/bin/python" \
  --pipeline_config /workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml \
  --sample_ids \
    val_bdd_pc_5k_video4166_phase3 \
    val_bdd_pc_5k_video6180_phase0 \
    val_bdd_pc_5k_video3207_phase4 \
    val_bdd_pc_5k_video5447_phase3 \
  --max_history_frames 41 \
  --prompt_mode behavior \
  --limit 4 \
  --resume

rm -f "$DENSE_MANIFEST"
rm -rf "$OUT/dense40_frames"
/opt/conda/envs/wan22/bin/python -m wan_ticft.data.materialize_future_manifest \
  --src_jsonl "$OUT/selected_manifest.jsonl" \
  --out_jsonl "$DENSE_MANIFEST" \
  --frames_root "$OUT/dense40_frames" \
  --future_frames 40

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$DENSE_MANIFEST" \
  --reports behavior="$OUT/report.csv" legacy_blank="$ROOT/ltxv_bdd25_gate/report.csv" \
  --offset_mode timestamp \
  --out_csv "${OUT}_dense40_eval.csv"

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$OUT/videos" \
  --mode videos \
  --out_dir "${OUT}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

echo
echo "Correction gate:"
echo "  - Pass only if behavior prompt improves macro PSNR or macro SSIM,"
echo "    does not create new actor hallucinations, and no sample loses >0.50 PSNR."
echo "  - If fail, retain legacy prompt and close LTX prompt tuning."
