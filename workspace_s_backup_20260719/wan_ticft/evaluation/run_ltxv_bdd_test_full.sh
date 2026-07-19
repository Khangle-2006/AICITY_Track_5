#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/ltxv_bdd_test_full
SMOKE=/workspace/experiments/wan_ticft/ltxv_bdd_test_smoke4
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 24000

# Reuse the four verified smoke outputs without duplicating their PNG data.
mkdir -p "$OUT"
for sample in video1170 video135 video1354 video1487; do
  if [[ -d "$SMOKE/$sample" && ! -e "$OUT/$sample" ]]; then
    cp -al "$SMOKE/$sample" "$OUT/$sample"
  fi
done

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.run_ltxv_test_bdd \
  --test_csv /workspace/experiments/index/test_index.csv \
  --out_dir "$OUT" \
  --ltx_repo /workspace/third_party/LTX-Video \
  --ltx_python /workspace/experiments/wan_ticft/ltxv_env/bin/python \
  --pipeline_config /workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml \
  --resume

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$OUT" \
  --mode frames \
  --out_dir "${OUT}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

echo "Inspect: ${OUT}_sheets"
