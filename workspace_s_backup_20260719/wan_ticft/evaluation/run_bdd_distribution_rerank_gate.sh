#!/usr/bin/env bash
set -euo pipefail

echo "=== BDD no-reference distribution rerank gate ==="
echo "Selects per-sample candidate by closeness to real-video Inception/R3D feature distribution."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export CUDA_VISIBLE_DEVICES
export PYTHONPATH=/workspace:/workspace/wan_ticft:${PYTHONPATH:-}

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
OUT=${OUT:-/workspace/experiments/wan_ticft/bdd_distribution_rerank_gate24}

"$PYTHON" -m wan_ticft.evaluation.rerank_distribution_router \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --report transition16=/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv \
  --report cnn0=/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix0/report.csv \
  --report mix45=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix45/report.csv \
  --report mix55=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix55/report.csv \
  --report mix60=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix60/report.csv \
  --report mix65=/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv \
  --report mix70=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix70/report.csv \
  --report mix75=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix75/report.csv \
  --clip_frames 16 \
  --frame_stride 4 \
  --image_weight 1.0 \
  --video_weight 1.0 \
  --smoothness_weight 0.05

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    transition16=/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv \
    cnn0=/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix0/report.csv \
    mix45=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix45/report.csv \
    mix65=/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv \
    mix75=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix75/report.csv \
    distribution_routed="$OUT/routed_report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report transition16=/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv \
  --report cnn0=/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix0/report.csv \
  --report mix45=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix45/report.csv \
  --report mix65=/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv \
  --report mix75=/workspace/experiments/wan_ticft/cnn_ltxmix_sweep_gate/mix75/report.csv \
  --report distribution_routed="$OUT/routed_report.csv" \
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
  - Continue rerank only if distribution_routed improves expected tradeoff
    versus fixed mix65 or mix45 on both pixel and proxy evidence.
  - If it just picks one constant candidate or worsens metrics, close this
    simple distribution reranker and try a learned selector with target-heldout.
EOF
