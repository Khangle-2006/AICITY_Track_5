#!/usr/bin/env bash
set -euo pipefail

cd /workspace
source /opt/conda/etc/profile.d/conda.sh
conda activate wan22

python -m wan_ticft.evaluation.audit_alignment_color \
  --manifest /workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl \
  --report /workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv \
  --out_csv /workspace/experiments/wan_ticft/ltxv_bdd25_alignment_color_audit.csv \
  --temporal_radius 3 \
  --spatial_radius 4
