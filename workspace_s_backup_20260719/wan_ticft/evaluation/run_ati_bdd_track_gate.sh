#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ATI=/workspace/third_party/ATI
INPUT=/workspace/experiments/wan_ticft/ati_bdd_smoke4
OUT=/workspace/experiments/wan_ticft/ati_bdd_smoke4_future_tracks

if [[ ! -d "$ATI" ]]; then
  echo "ATI repository is missing: $ATI" >&2
  exit 1
fi

rm -rf "$OUT"

echo "=== Extract observed motion and extrapolate future trajectories ==="
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.extract_ati_future_tracks \
  --ati_dir "$ATI" \
  --input_dir "$INPUT" \
  --out_dir "$OUT" \
  --num_points 40 \
  --future_frames 121 \
  --velocity_window 8 \
  --velocity_decay 0.99

echo "=== Render trajectory diagnostics ==="
/opt/conda/envs/wan22/bin/python "$ATI/tools/plot_user_inputs.py" \
  "$OUT/test.yaml" \
  --save_dir "$OUT/trajectory_images"

echo "=== Track gate result ==="
cat "$OUT/track_report.csv"
echo
echo "Inspect: $OUT/trajectory_images"
echo "Do not download ATI checkpoints until these four trajectory images pass visual review."
