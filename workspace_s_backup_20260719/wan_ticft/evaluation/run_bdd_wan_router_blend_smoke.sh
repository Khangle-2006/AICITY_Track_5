#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft/bdd_wan_router_blend_smoke4

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.build_frame_blends \
  --base_dir /workspace/experiments/wan_ticft/wan22_ti2v_test_121f_s10_frames \
  --correction_dir /workspace/experiments/wan_ticft/bdd_refiner_router_v2_sharp_smoke4 \
  --out_root "$ROOT" \
  --alphas 10 20 30 40 50 \
  --clean

for PERCENT in 10 20 30 40 50; do
  /opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
    --input "$ROOT/wan_router_${PERCENT}" \
    --mode frames \
    --out_dir "${ROOT}_sheets/wan_router_${PERCENT}" \
    --samples 12 \
    --tile_width 320 \
    --cols 4
done

