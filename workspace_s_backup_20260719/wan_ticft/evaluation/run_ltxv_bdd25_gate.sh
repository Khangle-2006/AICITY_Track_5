#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/ltxv_bdd25_gate
SOURCE_MANIFEST=/workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl
WAN_REPORT=/workspace/experiments/wan_ticft/wan22_bdd25_seed101_s10/report.csv
DENSE_MANIFEST="$OUT/dense40_manifest.jsonl"

GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 24000

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.run_ltxv_val_gate \
  --manifest "$SOURCE_MANIFEST" \
  --out_dir "$OUT" \
  --ltx_repo /workspace/third_party/LTX-Video \
  --ltx_python /workspace/experiments/wan_ticft/ltxv_env/bin/python \
  --pipeline_config /workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml \
  --limit 25 \
  --resume \
  --skip_bad

echo "=== Sparse manifest evaluation ==="
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports ltx="$OUT/report.csv" wan="$WAN_REPORT" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_sparse_eval.csv"

echo "=== Dense 40-frame evaluation ==="
rm -f "$DENSE_MANIFEST"
rm -rf "$OUT/dense40_frames"
/opt/conda/envs/wan22/bin/python -m wan_ticft.data.materialize_future_manifest \
  --src_jsonl "$OUT/selected_manifest.jsonl" \
  --out_jsonl "$DENSE_MANIFEST" \
  --frames_root "$OUT/dense40_frames" \
  --future_frames 40

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$DENSE_MANIFEST" \
  --reports ltx="$OUT/report.csv" wan="$WAN_REPORT" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_dense40_eval.csv"

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$OUT/videos" \
  --mode videos \
  --out_dir "${OUT}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4
