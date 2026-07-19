#!/usr/bin/env bash
set -euo pipefail

cd /workspace

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_ltx16_static_router \
  --manifest /workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl \
  --ltx16_report /workspace/experiments/wan_ticft/ltxv_bdd25_transitions/ltx_transition_16/report.csv \
  --static_fraction 0.20 \
  --out_csv /workspace/experiments/wan_ticft/ltx16_static20_router_gate.csv
