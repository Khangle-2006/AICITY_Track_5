#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl}
RAW_LTX=${RAW_LTX:-$ROOT/ltxv_bdd25_gate/report.csv}
TRANSITION16=${TRANSITION16:-$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv}
OUT=${OUT:-$ROOT/ltx_flow_bridge_gate}
PIXEL_CSV=${PIXEL_CSV:-$ROOT/ltx_flow_bridge_gate_eval.csv}
PROXY_CSV=${PROXY_CSV:-$ROOT/ltx_flow_bridge_gate_perceptual_proxy.csv}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== BDD LTX early motion bridge gate ==="
echo "Goal: remove first-frame transition ghosting without public-specific routing."

"$PYTHON" -m wan_ticft.evaluation.materialize_ltx_flow_bridge \
  --manifest "$MANIFEST" \
  --ltx_report "$RAW_LTX" \
  --out_root "$OUT" \
  --variant flow_bridge4_blend4=4,4 \
  --variant flow_bridge8_blend4=8,4 \
  --variant flow_bridge4_hard=4,0 \
  --clean

echo
echo "=== Pixel metrics ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    raw_ltx="$RAW_LTX" \
    transition16="$TRANSITION16" \
    flow_bridge4_blend4="$OUT/flow_bridge4_blend4/report.csv" \
    flow_bridge8_blend4="$OUT/flow_bridge8_blend4/report.csv" \
    flow_bridge4_hard="$OUT/flow_bridge4_hard/report.csv" \
  --offset_mode timestamp \
  --out_csv "$PIXEL_CSV"

echo
echo "=== Relative FID/FVD proxy ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report "transition16=$TRANSITION16" \
  --report "flow_bridge4_blend4=$OUT/flow_bridge4_blend4/report.csv" \
  --report "flow_bridge8_blend4=$OUT/flow_bridge8_blend4/report.csv" \
  --report "flow_bridge4_hard=$OUT/flow_bridge4_hard/report.csv" \
  --out_csv "$PROXY_CSV"

echo
echo "=== Contact sheets for visual ghosting QA ==="
"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/flow_bridge4_blend4/videos" \
  --mode videos \
  --out_dir "${OUT}/flow_bridge4_blend4_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4
"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/flow_bridge8_blend4/videos" \
  --mode videos \
  --out_dir "${OUT}/flow_bridge8_blend4_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4
"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/flow_bridge4_hard/videos" \
  --mode videos \
  --out_dir "${OUT}/flow_bridge4_hard_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

echo
echo "Gate rule:"
echo "  - Continue only if a flow bridge visibly reduces the first-frame ghost/double exposure."
echo "  - It must improve FVD proxy or lose less than 0.25 FVD versus transition16,"
echo "    and must not lose more than 0.15 PSNR / 0.005 SSIM."
echo "  - Prefer bridge4_blend4 over bridge8 if metrics are close, because shorter"
echo "    handoff is less likely to freeze LTX motion."
echo "  - If all fail, keep transition16 and close this bridge direction."
