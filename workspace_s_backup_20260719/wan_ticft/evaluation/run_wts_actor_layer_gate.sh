#!/usr/bin/env bash
set -euo pipefail

cd /workspace

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_manifest_motion_baselines \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --view_type wts_overhead_or_fixed_camera \
  --out_csv /workspace/experiments/wan_ticft/wts_fixed_actor_layer_gate.csv

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_manifest_motion_baselines \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --view_type wts_vehicle_or_ip_camera \
  --out_csv /workspace/experiments/wan_ticft/wts_ip_actor_layer_gate.csv
