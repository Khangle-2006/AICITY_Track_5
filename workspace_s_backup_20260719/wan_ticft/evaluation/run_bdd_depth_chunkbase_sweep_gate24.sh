#!/usr/bin/env bash
set -euo pipefail
# BDD24 gate: depth-layered with base = LTX2B CHUNK, sweeping fixed/adaptive/hdecay (+ optional DA3).
# Compares against the old-base a75 reference. Local gate only — verify against public separately.
cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=/workspace/experiments/wan_ticft
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl}
BASE_REPORT=${BASE_REPORT:-$ROOT/ltx2b_chunked_traffic_bdd24_c12/report.csv}   # base = ltx chunk
DEPTH_BACKEND=${DEPTH_BACKEND:-dpt}                                            # dpt | da3_mono | da3_metric
OUT=${OUT:-$ROOT/bdd_depth_chunkbase_sweep_${DEPTH_BACKEND}_eval24}
MODEL_CACHE=${MODEL_CACHE:-$ROOT/depth_layered_models/huggingface}
OLD_A75=${OLD_A75:-$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv}
CUDA_ID="${CUDA_VISIBLE_DEVICES:-2}"; CUDA_ID="${CUDA_ID%%,*}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== BDD24 depth-layered sweep | base=CHUNK | backend=$DEPTH_BACKEND ==="
echo "base_report=$BASE_REPORT"; echo "out=$OUT"
for p in "$MANIFEST" "$BASE_REPORT"; do [[ -s "$p" ]] || { echo "Missing: $p" >&2; exit 2; }; done

echo "[1/3] Render variants (a00 / a75 / adaptive_a75 / hdecay)"
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_root "$OUT" \
  --model_cache "$MODEL_CACHE" \
  --depth_backend "$DEPTH_BACKEND" \
  --view_type bdd \
  --limit 24 \
  --preserve_full_video \
  --alphas 0.0 0.75 \
  --adaptive_blend --adaptive_alpha 0.75 \
  --horizon_decay --horizon_decay_alpha 0.75 \
  --focal_scale 0.9 --point_stride 2 --splat_radius 1 \
  --min_matches 20 --min_inliers 12 --photo_sigma 32 \
  --segment_batch 4 --pose_smoothing_window 7 \
  --filled_pose_scale 0.35 --confidence_ema 0.65 \
  --clean

REPORTS=()
for v in depth_layered_a00 depth_layered_a75 depth_layered_adaptive_a75 depth_layered_hdecay; do
  [[ -s "$OUT/$v/report.csv" ]] && REPORTS+=( "${v}_chunk=$OUT/$v/report.csv" )
done
[[ -s "$OLD_A75" ]] && REPORTS+=( "a75_oldbase=$OLD_A75" )

echo "[2/3] Pixel metrics vs GT (sparse, timestamp offset)"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports "${REPORTS[@]}" \
  --include_repeat_last --offset_mode timestamp \
  --out_csv "$OUT/eval_pixel.csv"

echo "[3/3] FID/FVD proxy vs GT"
PROXY_ARGS=()
for r in "${REPORTS[@]}"; do PROXY_ARGS+=( --report "$r" ); done
CUDA_VISIBLE_DEVICES="$CUDA_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" "${PROXY_ARGS[@]}" \
  --clip_frames 8 --out_csv "$OUT/eval_proxy.csv"

echo
echo "=== DONE | backend=$DEPTH_BACKEND base=chunk ==="
echo "pixel=$OUT/eval_pixel.csv"
echo "proxy=$OUT/eval_proxy.csv"
echo "variants under: $OUT/<variant>/"
