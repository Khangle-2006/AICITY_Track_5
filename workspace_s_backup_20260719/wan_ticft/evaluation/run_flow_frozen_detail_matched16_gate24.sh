#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
TRAIN_ROOT="$ROOT/matched_cnn_ltxmix65_train16"
SOURCE_MANIFEST="$ROOT/manifests/train_ticft_balanced_300_frames.jsonl"
MATERIALIZED_MANIFEST="$TRAIN_ROOT/materialized_sparse_manifest.jsonl"
MATERIALIZED_FRAMES="$TRAIN_ROOT/materialized_sparse_frames"
LTX_OUT="$TRAIN_ROOT/ltx_raw"
TRAIN_MANIFEST="$TRAIN_ROOT/dense40_manifest.jsonl"
TRAIN_FRAMES="$TRAIN_ROOT/dense40_frames"
TRANSITION_ROOT="$TRAIN_ROOT/ltx_transitions"
TRANSITION_REPORT="$TRANSITION_ROOT/ltx_transition_16/report.csv"
SELECTOR_ROOT="$TRAIN_ROOT/cnn_selector"
TRAIN_REPORT="$TRAIN_ROOT/cnn_selector/cnn_ltxmix65/report.csv"
EVAL_MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
OUT="$ROOT/flow_frozen_detail_matched16_gate24"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Flow-frozen detail refiner: matched16 -> BDD24 ==="
echo "The base motion/geometry is frozen; only confidence-gated high-frequency residuals are learned."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 30000

for path in "$EVAL_MANIFEST" "$BASE_REPORT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing input: $path" >&2
    exit 2
  fi
done

if [[ ! -s "$TRAIN_MANIFEST" || ! -s "$TRAIN_REPORT" ]]; then
  echo
  echo "Matched16 dependency is missing; rebuilding it before refiner training."
  if [[ ! -f "$SOURCE_MANIFEST" ]]; then
    echo "Missing input: $SOURCE_MANIFEST" >&2
    exit 2
  fi
  if [[ ! -x "$ROOT/ltxv_env/bin/python" ]]; then
    echo "Missing LTX environment: $ROOT/ltxv_env/bin/python" >&2
    exit 2
  fi
  mkdir -p "$TRAIN_ROOT"

  MATERIALIZED_ROWS=0
  if [[ -s "$MATERIALIZED_MANIFEST" ]]; then
    MATERIALIZED_ROWS=$(wc -l < "$MATERIALIZED_MANIFEST")
  fi
  if (( MATERIALIZED_ROWS < 32 )); then
    echo "[prep 1/4] Materialize a 32-row BDD pool"
    "$PYTHON" -m wan_ticft.data.materialize_manifest_frames \
      --src_jsonl "$SOURCE_MANIFEST" \
      --out_jsonl "$MATERIALIZED_MANIFEST" \
      --frames_root "$MATERIALIZED_FRAMES" \
      --view_type bdd \
      --limit 32 \
      --clean
  else
    echo "[prep 1/4] Resume materialized pool: $MATERIALIZED_ROWS rows"
  fi

  echo "[prep 2/4] Generate/resume 16 LTX train predictions"
  "$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_gate \
    --manifest "$MATERIALIZED_MANIFEST" \
    --out_dir "$LTX_OUT" \
    --ltx_repo /workspace/third_party/LTX-Video \
    --ltx_python "$ROOT/ltxv_env/bin/python" \
    --pipeline_config /workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml \
    --view_type bdd \
    --limit 16 \
    --prompt_mode legacy \
    --resume \
    --skip_bad
  LTX_ROWS=$(( $(wc -l < "$LTX_OUT/report.csv") - 1 ))
  if (( LTX_ROWS != 16 )); then
    echo "Expected 16 completed LTX rows, found $LTX_ROWS." >&2
    exit 3
  fi

  echo "[prep 3/4] Materialize dense 40-frame targets"
  if [[ ! -s "$TRAIN_MANIFEST" ]]; then
    "$PYTHON" -m wan_ticft.data.materialize_future_manifest \
      --src_jsonl "$LTX_OUT/selected_manifest.jsonl" \
      --out_jsonl "$TRAIN_MANIFEST" \
      --frames_root "$TRAIN_FRAMES" \
      --future_frames 40
  else
    echo "resume dense manifest: $TRAIN_MANIFEST"
  fi

  echo "[prep 4/4] Build transition16 and out-of-fold cnn_ltxmix65 base"
  "$PYTHON" -m wan_ticft.evaluation.build_ltx_condition_transitions \
    --manifest "$TRAIN_MANIFEST" \
    --ltx_report "$LTX_OUT/report.csv" \
    --out_root "$TRANSITION_ROOT" \
    --transitions 16 \
    --clean
  "$PYTHON" -m wan_ticft.evaluation.materialize_cnn_ltx_region_selector_variants \
    --manifest "$TRAIN_MANIFEST" \
    --ltx_report "$TRANSITION_REPORT" \
    --out_root "$SELECTOR_ROOT" \
    --epochs 8 \
    --batch_size 8 \
    --variant cnn_ltxmix65:ltx_mix:0.65 \
    --clean
fi

for path in "$TRAIN_MANIFEST" "$TRAIN_REPORT"; do
  if [[ ! -s "$path" ]]; then
    echo "Matched dependency rebuild did not create: $path" >&2
    exit 3
  fi
done
TRAIN_ROWS=$(( $(wc -l < "$TRAIN_REPORT") - 1 ))
if (( TRAIN_ROWS != 16 )); then
  echo "Expected 16 matched-base rows, found $TRAIN_ROWS." >&2
  exit 3
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.train_flow_frozen_detail_refiner \
  --train_manifest "$TRAIN_MANIFEST" \
  --train_report "$TRAIN_REPORT" \
  --eval_manifest "$EVAL_MANIFEST" \
  --eval_report "$BASE_REPORT" \
  --out_dir "$OUT" \
  --train_limit 16 \
  --eval_limit 24 \
  --train_width 640 \
  --train_height 360 \
  --patch 160 \
  --batch_size 6 \
  --epochs 6 \
  --steps_per_epoch 80 \
  --model_width 48 \
  --residual_scale 0.035 \
  --highpass_sigma 2.0 \
  --highpass_kernel 9 \
  --consistency_px 2.5 \
  --clean

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_MANIFEST" \
  --reports cnn_ltxmix65="$BASE_REPORT" flow_frozen_detail="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_MANIFEST" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report flow_frozen_detail="$OUT/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

mapfile -t SAMPLE_IDS < <("$PYTHON" - <<PY
import json
for line in open("$EVAL_MANIFEST", encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        if row.get("view_type") == "bdd":
            print(row["sample_id"])
PY
)
SHEET_ARGS=()
for sample_id in "${SAMPLE_IDS[@]:0:24}"; do
  SHEET_ARGS+=(--sample_id "$sample_id")
done

"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$EVAL_MANIFEST" \
  --reports cnn_ltxmix65="$BASE_REPORT" flow_frozen_detail="$OUT/report.csv" \
  "${SHEET_ARGS[@]}" \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 220

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "diagnostics=$OUT/diagnostics.csv"
echo "sheets=${OUT}_compare_sheets"
echo
echo "Gate: continue only if PSNR improves >= 0.15, SSIM does not drop,"
echo "FID increase <= 0.15, FVD does not worsen, and sheets show no halos/crawl."
