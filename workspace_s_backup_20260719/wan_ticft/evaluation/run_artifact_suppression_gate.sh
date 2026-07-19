#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
MANIFEST=/workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl

echo "=== BDD selector alpha suppression ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_cnn_ltx_region_selector \
  --manifest /workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl \
  --ltx_report /workspace/experiments/wan_ticft/ltxv_bdd25_transitions/ltx_transition_16/report.csv \
  --out_csv /workspace/experiments/wan_ticft/cnn_artifact_suppression_gate.csv \
  --epochs 8 \
  --batch_size 8

echo "=== WTS opaque actor compositing ==="
for VIEW in wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera; do
  SHORT=${VIEW#wts_}
  "$PYTHON" -m wan_ticft.evaluation.evaluate_manifest_motion_baselines \
    --manifest "$MANIFEST" \
    --view_type "$VIEW" \
    --source_future_frames 50 \
    --out_csv "/workspace/experiments/wan_ticft/wts_${SHORT}_artifact_suppression_gate.csv"
done
