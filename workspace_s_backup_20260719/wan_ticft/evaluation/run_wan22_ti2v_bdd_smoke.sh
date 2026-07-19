#!/usr/bin/env bash
set -euo pipefail

echo "=== Wan2.2-TI2V-5B BDD image-anchor smoke ==="
echo "Model-level route: test whether Wan2.2 can beat cnn_ltxmix65 distribution without scene drift."

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
OUT=${OUT:-/workspace/experiments/wan_ticft/wan22_ti2v_bdd_smoke4}

"$PYTHON" -m wan_ticft.evaluation.run_wan22_ti2v_bdd_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --limit 4 \
  --width 832 \
  --height 480 \
  --num_frames 41 \
  --steps 10 \
  --seed 23 \
  --vram_limit 46

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    wan22_ti2v="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report wan22_ti2v="$OUT/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/videos" \
  --mode videos \
  --out_dir "${OUT}_sheets" \
  --samples 12 \
  --tile_width 320 \
  --cols 4

cat <<EOF

Outputs:
  videos: $OUT/videos
  eval: ${OUT}_eval.csv
  proxy: ${OUT}_proxy.csv
  sheets: ${OUT}_sheets

Gate rule:
  - Continue only if Wan2.2 keeps the same scene and improves FID/FVD proxy
    versus cnn_ltxmix65 or is visually much better with tolerable proxy.
  - If it changes scene/camera like Cosmos or loses both proxy and pixel, close
    Wan2.2-TI2V for this track.
EOF
