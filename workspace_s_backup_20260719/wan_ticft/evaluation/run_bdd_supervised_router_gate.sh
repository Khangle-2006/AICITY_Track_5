#!/usr/bin/env bash
set -euo pipefail

echo "=== BDD supervised held-out candidate router gate ==="
echo "Train on first 16 BDD rows, evaluate held-out behavior on remaining 8."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export CUDA_VISIBLE_DEVICES
export PYTHONPATH=/workspace:/workspace/wan_ticft:${PYTHONPATH:-}

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
OUT=${OUT:-/workspace/experiments/wan_ticft/bdd_supervised_router_gate24}

"$PYTHON" -m wan_ticft.evaluation.train_supervised_candidate_router \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --train_rows 16 \
  --default mix65 \
  --report cnn0=/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix0/report.csv \
  --report mix45=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix45/report.csv \
  --report mix55=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix55/report.csv \
  --report mix60=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix60/report.csv \
  --report mix65=/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv \
  --report mix70=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix70/report.csv \
  --report mix75=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix75/report.csv \
  --clean

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    cnn0=/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix0/report.csv \
    mix45=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix45/report.csv \
    mix65=/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv \
    mix75=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix75/report.csv \
    supervised_routed="$OUT/routed_report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report cnn0=/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix0/report.csv \
  --report mix45=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix45/report.csv \
  --report mix65=/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv \
  --report mix75=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix75/report.csv \
  --report supervised_routed="$OUT/routed_report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

cat <<EOF

Outputs:
  choices: $OUT/choices.csv
  routed: $OUT/routed_report.csv
  eval: ${OUT}_eval.csv
  proxy: ${OUT}_proxy.csv

Gate rule:
  - Continue only if held-out choices are not just overfit to cnn0 and aggregate
    score/proxy tradeoff beats fixed mix65/mix45.
  - If not, close supervised candidate routing.
EOF
