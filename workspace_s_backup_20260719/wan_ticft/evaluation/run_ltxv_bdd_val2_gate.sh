#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/ltxv_bdd_val2_gate
MANIFEST=/workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl

GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 24000

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.run_ltxv_val_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --ltx_repo /workspace/third_party/LTX-Video \
  --ltx_python /workspace/experiments/wan_ticft/ltxv_env/bin/python \
  --pipeline_config /workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml \
  --limit 2

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports ltx="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --limit 2 \
  --out_csv "${OUT}_eval.csv"

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$OUT/videos" \
  --mode videos \
  --out_dir "${OUT}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4
