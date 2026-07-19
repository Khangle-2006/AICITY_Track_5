#!/usr/bin/env bash
set -euo pipefail
# WTS val gate: measure which simple method is best PER WTS VIEW (overhead/fixed vs vehicle/ip).
# The frozen submission half (27/71) always used object-layer; this finally measures alternatives.
cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=/workspace/experiments/wan_ticft
VAL=${VAL:-$ROOT/manifests/val_ticft_balanced_125_frames.jsonl}
OUT=${OUT:-$ROOT/wts_methods_gate_val}
LIMIT=${LIMIT:-20}
CUDA_ID="${CUDA_VISIBLE_DEVICES:-2}"; CUDA_ID="${CUDA_ID%%,*}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

echo "=== WTS val method gate (per view) ==="; echo "val=$VAL out=$OUT limit_per_domain=$LIMIT"
[[ -s "$VAL" ]] || { echo "Missing $VAL" >&2; exit 2; }

echo "[1/3] Materialize GT target + candidate videos for both WTS views"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$VAL" --out_dir "$OUT" \
  --views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --limit_per_domain "$LIMIT" --cover_manifest_targets \
  --candidates repeat_last region_norm_medium region_norm_long12 region_norm_long24 \
  --clean

MAN="$OUT/selected_manifest.jsonl"
[[ -s "$MAN" ]] || MAN="$VAL"

echo "[2/3] Pixel metrics per candidate vs GT"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MAN" \
  --reports \
    repeat_last="$OUT/repeat_last_report.csv" \
    region_norm_medium="$OUT/region_norm_medium_report.csv" \
    region_norm_long12="$OUT/region_norm_long12_report.csv" \
    region_norm_long24="$OUT/region_norm_long24_report.csv" \
  --offset_mode timestamp \
  --out_csv "$OUT/eval_pixel.csv"

echo "[3/3] FID/FVD proxy per candidate vs GT"
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MAN" \
  --report repeat_last="$OUT/repeat_last_report.csv" \
  --report region_norm_medium="$OUT/region_norm_medium_report.csv" \
  --report region_norm_long12="$OUT/region_norm_long12_report.csv" \
  --report region_norm_long24="$OUT/region_norm_long24_report.csv" \
  --clip_frames 8 --out_csv "$OUT/eval_proxy.csv"

echo; echo "=== DONE ==="; echo "pixel=$OUT/eval_pixel.csv"; echo "proxy=$OUT/eval_proxy.csv"
