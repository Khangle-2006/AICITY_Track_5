#!/usr/bin/env bash
set -euo pipefail

echo "=== Wan/VACE generative refiner over cnn_ltxmix65 smoke ==="
echo "This is a breakthrough test, not a postprocess tweak."
echo "Runs 4 BDD clips with VACE control from cnn_ltxmix65 at two strengths."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export CUDA_VISIBLE_DEVICES
export PYTHONPATH=/workspace:/workspace/third_party/DiffSynth-Studio:/workspace/wan_ticft:${PYTHONPATH:-}
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=3600

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
BASE_REPORT=${BASE_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
OUT=${OUT:-/workspace/experiments/wan_ticft/vace_cnn_refiner_smoke4}

"$PYTHON" -m wan_ticft.evaluation.run_vace_cnn_refiner_gate \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_dir "$OUT" \
  --limit 4 \
  --width 832 \
  --height 480 \
  --num_frames 41 \
  --steps 8 \
  --seed 7 \
  --vace_scales 0.5,0.8

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    vace_s50="$OUT/vace_s50/report.csv" \
    vace_s80="$OUT/vace_s80/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report vace_s50="$OUT/vace_s50/report.csv" \
  --report vace_s80="$OUT/vace_s80/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

for label in vace_s50 vace_s80; do
  "$PYTHON" -m baseline.video_contact_sheet \
    --input "$OUT/$label/videos" \
    --mode videos \
    --out_dir "${OUT}_${label}_sheets" \
    --samples 12 \
    --tile_width 320 \
    --cols 4
done

cat <<EOF

Outputs:
  videos: $OUT
  eval: ${OUT}_eval.csv
  proxy: ${OUT}_proxy.csv
  sheets:
    ${OUT}_vace_s50_sheets
    ${OUT}_vace_s80_sheets

Gate rule:
  - Continue only if a VACE refiner clearly improves FID and FVD proxy versus
    cnn_ltxmix65 and visual QA keeps the same scene.
  - Pixel can drop slightly, but any scene replacement, cabin-view drift, or
    actor hallucination is a hard fail.
  - If both fail, close Wan/VACE refiner and move to another model-level route.
EOF
