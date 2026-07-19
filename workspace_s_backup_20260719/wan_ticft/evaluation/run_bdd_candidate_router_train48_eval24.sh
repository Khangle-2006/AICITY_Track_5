#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/experiments/wan_ticft
OUT=${OUT:-$ROOT/bdd_candidate_router_train45_eval24}
export PYTHONPATH=/workspace:${PYTHONPATH:-}
export OUT
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$ROOT/bdd_supervised_ltx_depth_train48/ltx_base/selected_manifest_repaired_conditions.jsonl}
EVAL_MANIFEST=${EVAL_MANIFEST:-$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl}
TRAIN_ROWS=${TRAIN_ROWS:-45}
EVAL_ROWS=${EVAL_ROWS:-24}
FRAMES=${FRAMES:-40}

TRAIN_DEPTH=${TRAIN_DEPTH:-$ROOT/bdd_supervised_ltx_depth_train48/depth/depth_layered_a75/report.csv}
EVAL_DEPTH=${EVAL_DEPTH:-$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv}
TRAIN_DEPTH_A00=${TRAIN_DEPTH_A00:-$ROOT/bdd_supervised_ltx_depth_train48/depth/depth_layered_a00/report.csv}
EVAL_DEPTH_A00=${EVAL_DEPTH_A00:-$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a00/report.csv}
TRAIN_LTX=${TRAIN_LTX:-$ROOT/bdd_supervised_ltx_depth_train48/ltx_base/report.csv}
TRAIN_REPEAT=${TRAIN_REPEAT:-$ROOT/bdd_supervised_repeat_depth_train48/repeat_last/report.csv}
EVAL_REPEAT=${EVAL_REPEAT:-$ROOT/bdd_supervised_repeat_depth_eval24/repeat_last/report.csv}

echo "=== BDD candidate router train -> eval24 ==="
echo "out=$OUT"
echo "train_manifest=$TRAIN_MANIFEST"
echo "eval_manifest=$EVAL_MANIFEST"

rm -rf "$OUT"
mkdir -p "$OUT/combined"

echo "[1/8] Materialize train candidate pool"
python -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$TRAIN_MANIFEST" \
  --out_dir "$OUT/train_pool" \
  --views bdd \
  --limit_per_domain "$TRAIN_ROWS" \
  --frames "$FRAMES" \
  --cover_manifest_targets \
  --candidates region_norm_medium region_norm_long24 actor_layer_small_filled \
  --clean

echo "[2/8] Materialize eval candidate pool"
python -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$EVAL_MANIFEST" \
  --out_dir "$OUT/eval_pool" \
  --views bdd \
  --limit_per_domain "$EVAL_ROWS" \
  --frames "$FRAMES" \
  --cover_manifest_targets \
  --candidates region_norm_medium region_norm_long24 actor_layer_small_filled \
  --clean

echo "[3/8] Combine manifests and candidate reports"
python -m wan_ticft.evaluation.combine_jsonl_manifests \
  --input "$TRAIN_MANIFEST" \
  --input "$EVAL_MANIFEST" \
  --out "$OUT/combined/manifest_train_eval24.jsonl"

python -m wan_ticft.evaluation.combine_video_reports \
  --input "$TRAIN_DEPTH" \
  --input "$EVAL_DEPTH" \
  --out "$OUT/combined/depth_a75_report.csv"

python -m wan_ticft.evaluation.combine_video_reports \
  --input "$TRAIN_DEPTH_A00" \
  --input "$EVAL_DEPTH_A00" \
  --out "$OUT/combined/depth_a00_report.csv"

python -m wan_ticft.evaluation.combine_video_reports \
  --input "$TRAIN_REPEAT" \
  --input "$EVAL_REPEAT" \
  --out "$OUT/combined/repeat_last_report.csv"

python -m wan_ticft.evaluation.combine_video_reports \
  --input "$OUT/train_pool/region_norm_medium_report.csv" \
  --input "$OUT/eval_pool/region_norm_medium_report.csv" \
  --out "$OUT/combined/region_medium_report.csv"

python -m wan_ticft.evaluation.combine_video_reports \
  --input "$OUT/train_pool/region_norm_long24_report.csv" \
  --input "$OUT/eval_pool/region_norm_long24_report.csv" \
  --out "$OUT/combined/region_long24_report.csv"

python -m wan_ticft.evaluation.combine_video_reports \
  --input "$OUT/train_pool/actor_layer_small_filled_report.csv" \
  --input "$OUT/eval_pool/actor_layer_small_filled_report.csv" \
  --out "$OUT/combined/actor_small_report.csv"

if [[ -f "$TRAIN_LTX" ]]; then
  python -m wan_ticft.evaluation.combine_video_reports \
    --input "$TRAIN_LTX" \
    --input "$EVAL_DEPTH" \
    --out "$OUT/combined/ltx_or_depth_fallback_report.csv"
fi

echo "[4/8] Train supervised candidate router on first $TRAIN_ROWS rows"
python -m wan_ticft.evaluation.train_supervised_candidate_router \
  --manifest "$OUT/combined/manifest_train_eval24.jsonl" \
  --report depth_a75="$OUT/combined/depth_a75_report.csv" \
  --report depth_a00="$OUT/combined/depth_a00_report.csv" \
  --report repeat_last="$OUT/combined/repeat_last_report.csv" \
  --report region_medium="$OUT/combined/region_medium_report.csv" \
  --report region_long24="$OUT/combined/region_long24_report.csv" \
  --report actor_small="$OUT/combined/actor_small_report.csv" \
  --out_dir "$OUT/router" \
  --train_rows "$TRAIN_ROWS" \
  --default depth_a75 \
  --clean

echo "[5/8] Build eval oracle upper bound"
python -m wan_ticft.evaluation.materialize_oracle_report \
  --manifest "$EVAL_MANIFEST" \
  --reports \
    depth_a75="$OUT/combined/depth_a75_report.csv" \
    depth_a00="$OUT/combined/depth_a00_report.csv" \
    repeat_last="$OUT/combined/repeat_last_report.csv" \
    region_medium="$OUT/combined/region_medium_report.csv" \
    region_long24="$OUT/combined/region_long24_report.csv" \
    actor_small="$OUT/combined/actor_small_report.csv" \
  --out_dir "$OUT/eval_oracle_absdiff" \
  --metric absdiff \
  --offset_mode timestamp

echo "[6/8] Pixel metrics on held-out eval24 only"
python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/combined/manifest_train_eval24.jsonl" \
  --offset "$TRAIN_ROWS" \
  --limit "$EVAL_ROWS" \
  --reports \
    depth_a75="$OUT/combined/depth_a75_report.csv" \
    depth_a00="$OUT/combined/depth_a00_report.csv" \
    repeat_last="$OUT/combined/repeat_last_report.csv" \
    region_medium="$OUT/combined/region_medium_report.csv" \
    region_long24="$OUT/combined/region_long24_report.csv" \
    actor_small="$OUT/combined/actor_small_report.csv" \
    router="$OUT/router/routed_report.csv" \
    oracle="$OUT/eval_oracle_absdiff/oracle_report.csv" \
  --offset_mode timestamp \
  --out_csv "$OUT/eval24_pixel.csv"

echo "[7/8] FID/FVD proxy on held-out eval24"
python -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_MANIFEST" \
  --report depth_a75="$OUT/combined/depth_a75_report.csv" \
  --report region_medium="$OUT/combined/region_medium_report.csv" \
  --report region_long24="$OUT/combined/region_long24_report.csv" \
  --report actor_small="$OUT/combined/actor_small_report.csv" \
  --report router="$OUT/router/routed_report.csv" \
  --report oracle="$OUT/eval_oracle_absdiff/oracle_report.csv" \
  --clip_frames 8 \
  --out_csv "$OUT/eval24_proxy.csv"

echo "[8/8] Choice summary"
python - <<'PY'
from collections import Counter
import csv
import os
from pathlib import Path
out = Path(os.environ["OUT"])
choices = out / "router" / "choices.csv"
rows = list(csv.DictReader(choices.open()))
for split in ("train", "eval"):
    part = [r for r in rows if r["split"] == split]
    print(split, "chosen", dict(Counter(r["chosen"] for r in part)))
    print(split, "oracle", dict(Counter(r["oracle"] for r in part)))
print(f"choices={choices}")
PY

echo
echo "pixel=$OUT/eval24_pixel.csv"
echo "proxy=$OUT/eval24_proxy.csv"
echo "choices=$OUT/router/choices.csv"
echo
echo "Gate: continue only if router beats depth_a75 on eval24 proxy and does not lose pixel metrics badly."
