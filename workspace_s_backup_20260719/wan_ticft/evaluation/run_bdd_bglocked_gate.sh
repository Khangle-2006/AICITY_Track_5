#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl}
DEPTH_REPORT=${DEPTH_REPORT:-$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv}
LIMIT_PER_DOMAIN=${LIMIT_PER_DOMAIN:-24}
OUT=${OUT:-$ROOT/bdd_bglocked_gate${LIMIT_PER_DOMAIN}}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== BDD background-locked actor composite gate ==="
echo "out=$OUT"
echo "manifest=$MANIFEST"
echo "limit_per_domain=$LIMIT_PER_DOMAIN"

for path in "$MANIFEST"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

rm -rf "$OUT"

echo
echo "[1/3] Materialize BDD candidates"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --limit_per_domain "$LIMIT_PER_DOMAIN" \
  --views bdd \
  --frames 50 \
  --cover_manifest_targets \
  --candidates \
    region_norm_medium \
    region_norm_long24 \
    actor_layer_small_filled \
    bg_locked_region_long24 \
    bg_locked_actor_small \
  --clean

echo
echo "[2/3] Pixel metrics"
REPORT_ARGS=(
  --report "region_medium=$OUT/region_norm_medium_report.csv"
  --report "region_long24=$OUT/region_norm_long24_report.csv"
  --report "actor_small=$OUT/actor_layer_small_filled_report.csv"
  --report "bg_locked_long24=$OUT/bg_locked_region_long24_report.csv"
  --report "bg_locked_actor_small=$OUT/bg_locked_actor_small_report.csv"
)
FULL_VIDEO_REPORT_ARGS=("${REPORT_ARGS[@]}")
if [[ -s "$DEPTH_REPORT" ]]; then
  REPORT_ARGS+=(--report "depth_a75=$DEPTH_REPORT")
fi
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_report_pixels \
  --target_report "$OUT/target_report.csv" \
  "${REPORT_ARGS[@]}" \
  --out_csv "$ROOT/bdd_bglocked_gate${LIMIT_PER_DOMAIN}_pixel.csv"

echo
echo "[3/4] FID/FVD proxy on full materialized videos"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --target_report "$OUT/target_report.csv" \
  --full_video \
  "${FULL_VIDEO_REPORT_ARGS[@]}" \
  --out_csv "$ROOT/bdd_bglocked_gate${LIMIT_PER_DOMAIN}_proxy.csv"

echo
echo "[4/4] FID/FVD proxy on manifest target frames"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  "${REPORT_ARGS[@]}" \
  --clip_frames 8 \
  --frame_stride 1 \
  --out_csv "$ROOT/bdd_bglocked_gate${LIMIT_PER_DOMAIN}_proxy_timestamp.csv"

echo
echo "Gate rule:"
echo "  - Continue only if bg_locked_actor_small improves FID/FVD versus actor_small"
echo "    and remains competitive with depth_a75."
echo "  - If it loses badly to depth_a75, do not apply the green-screen route to BDD."
echo
echo "pixel=$ROOT/bdd_bglocked_gate${LIMIT_PER_DOMAIN}_pixel.csv"
echo "proxy=$ROOT/bdd_bglocked_gate${LIMIT_PER_DOMAIN}_proxy.csv"
echo "proxy_timestamp=$ROOT/bdd_bglocked_gate${LIMIT_PER_DOMAIN}_proxy_timestamp.csv"
