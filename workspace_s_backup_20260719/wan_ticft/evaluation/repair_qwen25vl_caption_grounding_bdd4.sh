#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
OUT="$ROOT/qwen25vl_caption_grounding_bdd4"

echo "=== Repair existing Qwen2.5-VL BDD4 responses without loading the model ==="
"$PYTHON" -m wan_ticft.evaluation.run_qwen25vl_caption_grounding \
  --manifest "$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl" \
  --out_dir "$OUT" \
  --sample_id val_bdd_pc_5k_video2556_phase2 \
  --sample_id val_bdd_pc_5k_video8125_phase1 \
  --sample_id val_bdd_pc_5k_video8139_phase2 \
  --sample_id val_bdd_pc_5k_video8277_phase2 \
  --limit 4 \
  --reuse_responses

echo
echo "report=$OUT/report.csv"
echo "sheets=$OUT/*/grounding_sheet.jpg"
