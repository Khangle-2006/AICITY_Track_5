#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
OUT=/workspace/experiments/wan_ticft/cnn_sharp2_bdd_test

"$PYTHON" -m wan_ticft.evaluation.train_apply_cnn_ltx_region_selector \
  --manifest /workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl \
  --ltx_report /workspace/experiments/wan_ticft/ltxv_bdd25_transitions/ltx_transition_16/report.csv \
  --test_csv /workspace/experiments/index/test_index.csv \
  --ltx_test_dir /workspace/experiments/wan_ticft/ltxv_bdd_test_transition16 \
  --region_test_dir /workspace/experiments/region_norm_medium_submission \
  --out_dir "$OUT" \
  --checkpoint /workspace/experiments/wan_ticft/cnn_sharp2_final.pt \
  --report_csv /workspace/experiments/wan_ticft/cnn_sharp2_bdd_test_report.csv \
  --epochs 10 \
  --batch_size 8 \
  --alpha_strength 2.0 \
  --clean

"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT" \
  --mode frames \
  --out_dir "${OUT}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

echo "Compare:"
echo "  soft:  /workspace/experiments/wan_ticft/cnn_ltx_region_selector_bdd_test_sheets"
echo "  sharp: ${OUT}_sheets"
