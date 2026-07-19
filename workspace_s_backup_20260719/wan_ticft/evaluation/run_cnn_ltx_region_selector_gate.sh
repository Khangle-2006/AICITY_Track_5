#!/usr/bin/env bash
set -euo pipefail

cd /workspace

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_cnn_ltx_region_selector \
  --manifest /workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl \
  --ltx_report /workspace/experiments/wan_ticft/ltxv_bdd25_transitions/ltx_transition_16/report.csv \
  --out_csv /workspace/experiments/wan_ticft/cnn_ltx_region_selector_gate.csv \
  --epochs 8 \
  --batch_size 8
