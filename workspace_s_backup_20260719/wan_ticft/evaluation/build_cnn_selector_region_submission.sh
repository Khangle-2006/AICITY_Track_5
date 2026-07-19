#!/usr/bin/env bash
set -euo pipefail

cd /workspace

BDD_OUT=/workspace/experiments/wan_ticft/cnn_ltx_region_selector_bdd_test
CHECKPOINT=/workspace/experiments/wan_ticft/cnn_ltx_region_selector_final.pt
SUBMISSION=/workspace/experiments/hybrid_cnn_selector_region_wts_submission

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.train_apply_cnn_ltx_region_selector \
  --manifest /workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl \
  --ltx_report /workspace/experiments/wan_ticft/ltxv_bdd25_transitions/ltx_transition_16/report.csv \
  --test_csv /workspace/experiments/index/test_index.csv \
  --ltx_test_dir /workspace/experiments/wan_ticft/ltxv_bdd_test_transition16 \
  --region_test_dir /workspace/experiments/region_norm_medium_submission \
  --out_dir "$BDD_OUT" \
  --checkpoint "$CHECKPOINT" \
  --report_csv /workspace/experiments/wan_ticft/cnn_ltx_region_selector_bdd_test_report.csv \
  --epochs 10 \
  --batch_size 8 \
  --clean

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$BDD_OUT" \
  --mode frames \
  --out_dir "${BDD_OUT}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir /workspace/experiments/index \
  --wan_dir "$BDD_OUT" \
  --wan_label cnn_ltx_region_selector \
  --region_dir /workspace/experiments/region_norm_medium_submission \
  --region_label region_norm_medium \
  --out_dir "$SUBMISSION" \
  --report_csv /workspace/experiments/hybrid_cnn_selector_region_wts_verify.csv \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

/opt/conda/envs/wan22/bin/python -m baseline.submission \
  --index_dir /workspace/experiments/index \
  --gen_dir "$SUBMISSION" \
  --submission_dir /workspace/experiments/hybrid_cnn_selector_region_wts_submission_verified \
  --report_csv /workspace/experiments/hybrid_cnn_selector_region_wts_submission_verify.csv \
  --verify_only

echo "Inspect before packaging: ${BDD_OUT}_sheets"
