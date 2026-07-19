#!/usr/bin/env bash
set -euo pipefail

cd /workspace

manifest=/workspace/experiments/wan_ticft/manifests/val_ticft_bdd12_matched_121f_v2.jsonl
frames_root=/workspace/experiments/wan_ticft/frames_val_bdd12_matched_121f_v2
out_csv=/workspace/experiments/wan_ticft/wan22_multiseed_bdd12_eval_matched_121f_v2.csv

python -m wan_ticft.data.materialize_future_manifest \
  --src_jsonl /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --out_jsonl "$manifest" \
  --frames_root "$frames_root" \
  --future_frames 120 \
  --limit 12

python -m wan_ticft.data.verify_manifest \
  --jsonl "$manifest" \
  --require_frames \
  --require_target

python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$manifest" \
  --reports \
    seed101=/workspace/experiments/wan_ticft/wan22_multiseed_bdd12_seed101/report.csv \
    seed1001=/workspace/experiments/wan_ticft/wan22_multiseed_bdd12_seed1001/report.csv \
    lora_a07=/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12_a07/report.csv \
  --include_repeat_last \
  --out_csv "$out_csv"

echo "manifest: $manifest"
echo "evaluation: $out_csv"
