#!/usr/bin/env bash
set -euo pipefail

cd /workspace

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_ltx_region_composite_ceiling \
  --manifest /workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl \
  --ltx_report /workspace/experiments/wan_ticft/ltxv_bdd25_transitions/ltx_transition_16/report.csv \
  --out_csv /workspace/experiments/wan_ticft/ltx_region_composite_ceiling.csv
