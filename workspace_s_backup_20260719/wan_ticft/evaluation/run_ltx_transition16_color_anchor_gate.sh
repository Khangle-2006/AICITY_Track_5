#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft/ltxv_bdd25_transition16_color_anchor
MANIFEST=/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl
LTX_REPORT=/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv
LINEAR_REPORT=/workspace/experiments/wan_ticft/ltxv_bdd25_transitions/ltx_transition_16/report.csv

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.build_ltx_condition_transitions \
  --manifest "$MANIFEST" \
  --ltx_report "$LTX_REPORT" \
  --out_root "$ROOT" \
  --transitions 16 \
  --color_anchor_strength 0.25 \
  --clean

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    linear16="$LINEAR_REPORT" \
    color_anchor25="$ROOT/ltx_transition_16_color25/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "$ROOT/eval.csv"

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$ROOT/ltx_transition_16_color25/videos" \
  --mode videos \
  --out_dir "$ROOT/ltx_transition_16_color25_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4
