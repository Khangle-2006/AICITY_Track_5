#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/ltxv_bdd_dense6_eval

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.prepare_ltxv_dense6_eval

rm -f "$OUT/dense40_manifest.jsonl"
rm -rf "$OUT/frames"

/opt/conda/envs/wan22/bin/python -m wan_ticft.data.materialize_future_manifest \
  --src_jsonl "$OUT/source_manifest.jsonl" \
  --out_jsonl "$OUT/dense40_manifest.jsonl" \
  --frames_root "$OUT/frames" \
  --future_frames 40

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/dense40_manifest.jsonl" \
  --reports \
    ltx="$OUT/ltx_report.csv" \
    wan=/workspace/experiments/wan_ticft/wan22_bdd25_seed101_s10/report.csv \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "$OUT/eval.csv"
