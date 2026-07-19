#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
TRANSITION_ROOT="$ROOT/ltxv_bdd25_transition_duration_gate"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== BDD transition-duration proxy by phase ==="
echo "Use this only for phase-level routing; do not route by validation sample ID."

for PHASE in 0 1 2 3 4; do
  echo
  echo "=== phase $PHASE ==="
  /workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  TORCH_HOME="$ROOT/tools/torch_cache" \
  "$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
    --manifest "$MANIFEST" \
    --phase "$PHASE" \
    --report "transition16=$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv" \
    --report "transition24=$TRANSITION_ROOT/ltx_transition_24/report.csv" \
    --report "transition32=$TRANSITION_ROOT/ltx_transition_32/report.csv" \
    --out_csv "$ROOT/bdd_transition_phase${PHASE}_perceptual_proxy.csv"
done

echo
echo "Gate rule:"
echo "  - Consider phase routing only for phases where transition24/32 improves"
echo "    both FID and FVD proxies versus transition16."
echo "  - Then run one aggregate routed proxy gate before building test outputs."
echo "  - If no phase passes both proxies, keep transition16 and close BDD"
echo "    transition routing."
