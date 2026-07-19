#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/ltxv_bdd_future_anchor4_gate
SOURCE=/workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl
DENSE="$OUT/dense40_manifest.jsonl"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 24000

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.run_ltxv_val_gate \
  --manifest "$SOURCE" \
  --out_dir "$OUT" \
  --ltx_repo /workspace/third_party/LTX-Video \
  --ltx_python /workspace/experiments/wan_ticft/ltxv_env/bin/python \
  --pipeline_config /workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml \
  --view_type bdd \
  --phases 0 1 2 3 \
  --limit 4 \
  --future_anchor_offset 16 \
  --future_anchor_strength 0.25 \
  --resume

rm -f "$DENSE"
rm -rf "$OUT/dense40_frames"
/opt/conda/envs/wan22/bin/python -m wan_ticft.data.materialize_future_manifest \
  --src_jsonl "$OUT/selected_manifest.jsonl" \
  --out_jsonl "$DENSE" \
  --frames_root "$OUT/dense40_frames" \
  --future_frames 40

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$DENSE" \
  --reports \
    anchor="$OUT/report.csv" \
    ltx=/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv \
    ltx16=/workspace/experiments/wan_ticft/ltxv_bdd25_transitions/ltx_transition_16/report.csv \
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
