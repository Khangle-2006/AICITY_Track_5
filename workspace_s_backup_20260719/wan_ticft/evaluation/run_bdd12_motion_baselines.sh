#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python -m wan_ticft.evaluation.evaluate_manifest_motion_baselines \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_bdd12_matched_121f_v2.jsonl \
  --out_csv /workspace/experiments/wan_ticft/bdd12_matched_121f_motion_baselines.csv
