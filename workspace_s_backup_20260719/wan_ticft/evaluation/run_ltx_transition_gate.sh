#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft/ltxv_bdd25_transitions
MANIFEST=/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl
LTX_REPORT=/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.build_ltx_condition_transitions \
  --manifest "$MANIFEST" \
  --ltx_report "$LTX_REPORT" \
  --out_root "$ROOT" \
  --transitions 8 16 \
  --clean

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    ltx="$LTX_REPORT" \
    transition8="$ROOT/ltx_transition_8/report.csv" \
    transition16="$ROOT/ltx_transition_16/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "$ROOT/eval.csv"

for transition in 8 16; do
  /opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
    --input "$ROOT/ltx_transition_${transition}/videos" \
    --mode videos \
    --out_dir "$ROOT/ltx_transition_${transition}_sheets" \
    --samples 16 \
    --tile_width 320 \
    --cols 4
done
