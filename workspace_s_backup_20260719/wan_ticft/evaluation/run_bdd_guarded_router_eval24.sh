#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/experiments/wan_ticft
OUT=${OUT:-$ROOT/bdd_candidate_router_train45_eval24}
EVAL_MANIFEST=${EVAL_MANIFEST:-$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl}
export PYTHONPATH=/workspace:${PYTHONPATH:-}

echo "=== BDD guarded router eval24 ==="
echo "out=$OUT"

python -m wan_ticft.evaluation.build_guarded_router_reports \
  --manifest "$EVAL_MANIFEST" \
  --choices "$OUT/router/choices.csv" \
  --report depth_a75="$OUT/combined/depth_a75_report.csv" \
  --report repeat_last="$OUT/combined/repeat_last_report.csv" \
  --report region_medium="$OUT/combined/region_medium_report.csv" \
  --report region_long24="$OUT/combined/region_long24_report.csv" \
  --report actor_small="$OUT/combined/actor_small_report.csv" \
  --out_dir "$OUT/guarded"

python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_MANIFEST" \
  --reports \
    depth_a75="$OUT/combined/depth_a75_report.csv" \
    repeat_last="$OUT/combined/repeat_last_report.csv" \
    region_medium="$OUT/combined/region_medium_report.csv" \
    region_long24="$OUT/combined/region_long24_report.csv" \
    actor_small="$OUT/combined/actor_small_report.csv" \
    router="$OUT/router/routed_report.csv" \
    router_medium_guard="$OUT/guarded/router_medium_guard_report.csv" \
    phase0_depth_router_medium_guard="$OUT/guarded/phase0_depth_router_medium_guard_report.csv" \
    phase0_depth_phase12_repeat_router_medium_guard="$OUT/guarded/phase0_depth_phase12_repeat_router_medium_guard_report.csv" \
  --offset_mode timestamp \
  --out_csv "$OUT/guarded_eval24_pixel.csv"

python -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_MANIFEST" \
  --report depth_a75="$OUT/combined/depth_a75_report.csv" \
  --report repeat_last="$OUT/combined/repeat_last_report.csv" \
  --report region_medium="$OUT/combined/region_medium_report.csv" \
  --report region_long24="$OUT/combined/region_long24_report.csv" \
  --report actor_small="$OUT/combined/actor_small_report.csv" \
  --report router="$OUT/router/routed_report.csv" \
  --report router_medium_guard="$OUT/guarded/router_medium_guard_report.csv" \
  --report phase0_depth_router_medium_guard="$OUT/guarded/phase0_depth_router_medium_guard_report.csv" \
  --report phase0_depth_phase12_repeat_router_medium_guard="$OUT/guarded/phase0_depth_phase12_repeat_router_medium_guard_report.csv" \
  --clip_frames 8 \
  --out_csv "$OUT/guarded_eval24_proxy.csv"

echo
echo "pixel=$OUT/guarded_eval24_pixel.csv"
echo "proxy=$OUT/guarded_eval24_proxy.csv"
