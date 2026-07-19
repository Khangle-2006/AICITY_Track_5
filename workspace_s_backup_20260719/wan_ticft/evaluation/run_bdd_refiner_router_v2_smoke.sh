#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/bdd_refiner_router_v2_sharp_smoke4

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
/opt/conda/envs/wan22/bin/python -m wan_ticft.inference.run_bdd_refiner_router_v2 \
  --index_dir /workspace/experiments/index \
  --checkpoint /workspace/experiments/refiner_bdd_direct_h1_120_v2/refiner_best.pt \
  --out "$OUT" \
  --limit 4 \
  --clean

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$OUT" \
  --mode frames \
  --out_dir "${OUT}_sheets" \
  --samples 12 \
  --tile_width 320 \
  --cols 4
