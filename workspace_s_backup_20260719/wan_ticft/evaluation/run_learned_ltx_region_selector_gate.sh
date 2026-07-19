#!/usr/bin/env bash
set -euo pipefail

cd /workspace

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_learned_ltx_region_selector \
  --manifest /workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl \
  --ltx_report /workspace/experiments/wan_ticft/ltxv_bdd25_transitions/ltx_transition_16/report.csv \
  --out_csv /workspace/experiments/wan_ticft/learned_ltx_region_selector_gate.csv \
  --block 64
