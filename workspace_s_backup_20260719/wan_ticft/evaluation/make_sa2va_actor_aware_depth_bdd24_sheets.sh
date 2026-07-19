#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
OUT="$ROOT/sa2va_actor_aware_depth_bdd24"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
DEPTH_REPORT="$ROOT/depth_layered_renderer_temporal_bdd24/depth_layered_a75/report.csv"
SHEETS="${OUT}_compare_sheets"

for path in \
  "$OUT/selected_manifest.jsonl" \
  "$BASE_REPORT" \
  "$DEPTH_REPORT" \
  "$OUT/actor_depth_a90/report.csv" \
  "$OUT/actor_depth_a100/report.csv"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

rm -rf "$SHEETS"
"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    depth_a75="$DEPTH_REPORT" \
    actor_depth_a90="$OUT/actor_depth_a90/report.csv" \
    actor_depth_a100="$OUT/actor_depth_a100/report.csv" \
  --sample_id val_bdd_pc_5k_video7040_phase0 \
  --sample_id val_bdd_pc_5k_video1407_phase0 \
  --sample_id val_bdd_pc_5k_video8277_phase2 \
  --sample_id val_bdd_pc_5k_video8139_phase2 \
  --sample_id val_bdd_pc_5k_video6075_phase3 \
  --sample_id val_bdd_pc_5k_video4166_phase3 \
  --sample_id val_bdd_pc_5k_video9450_phase4 \
  --sample_id val_bdd_pc_5k_video3207_phase4 \
  --out_dir "$SHEETS" \
  --frames 8 \
  --tile_width 220

echo "sheets=$SHEETS"
