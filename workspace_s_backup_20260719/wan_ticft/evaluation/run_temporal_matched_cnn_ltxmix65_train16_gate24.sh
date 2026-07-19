#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-$ROOT/manifests/train_ticft_balanced_300_frames.jsonl}
EVAL_MANIFEST=${EVAL_MANIFEST:-$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl}
EVAL_BASE_REPORT=${EVAL_BASE_REPORT:-$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
TRAIN_LIMIT=${TRAIN_LIMIT:-16}
POOL_LIMIT=${POOL_LIMIT:-32}

TRAIN_ROOT="$ROOT/matched_cnn_ltxmix65_train${TRAIN_LIMIT}"
MATERIALIZED_MANIFEST="$TRAIN_ROOT/materialized_sparse_manifest.jsonl"
MATERIALIZED_FRAMES="$TRAIN_ROOT/materialized_sparse_frames"
LTX_OUT="$TRAIN_ROOT/ltx_raw"
DENSE_MANIFEST="$TRAIN_ROOT/dense40_manifest.jsonl"
DENSE_FRAMES="$TRAIN_ROOT/dense40_frames"
TRANSITION_ROOT="$TRAIN_ROOT/ltx_transitions"
TRANSITION_REPORT="$TRANSITION_ROOT/ltx_transition_16/report.csv"
MATCHED_BASE_ROOT="$TRAIN_ROOT/cnn_selector"
MATCHED_BASE_REPORT="$MATCHED_BASE_ROOT/cnn_ltxmix65/report.csv"
OUT="$ROOT/temporal_matched_cnn_ltxmix65_train${TRAIN_LIMIT}_gate24"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Matched cnn_ltxmix65 refiner: train${TRAIN_LIMIT} -> BDD24 ==="
echo "The full train manifest contains only 60 BDD rows; this is a bounded train16 gate."
echo "train_root=$TRAIN_ROOT"
echo "out=$OUT"
echo "gpu=$GPU_ID"

for path in "$SOURCE_MANIFEST" "$EVAL_MANIFEST" "$EVAL_BASE_REPORT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing input: $path" >&2
    exit 2
  fi
done
if [[ ! -x "$ROOT/ltxv_env/bin/python" ]]; then
  echo "Missing LTX environment: $ROOT/ltxv_env/bin/python" >&2
  exit 2
fi
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 30000
mkdir -p "$TRAIN_ROOT"

echo
echo "[1/7] Materialize a ${POOL_LIMIT}-row BDD candidate pool for ${TRAIN_LIMIT} valid outputs"
MATERIALIZED_ROWS=0
if [[ -s "$MATERIALIZED_MANIFEST" ]]; then
  MATERIALIZED_ROWS=$(wc -l < "$MATERIALIZED_MANIFEST")
fi
if (( MATERIALIZED_ROWS < POOL_LIMIT )); then
  "$PYTHON" -m wan_ticft.data.materialize_manifest_frames \
    --src_jsonl "$SOURCE_MANIFEST" \
    --out_jsonl "$MATERIALIZED_MANIFEST" \
    --frames_root "$MATERIALIZED_FRAMES" \
    --view_type bdd \
    --limit "$POOL_LIMIT" \
    --clean
else
  echo "resume materialized manifest: $MATERIALIZED_MANIFEST ($MATERIALIZED_ROWS rows)"
fi

echo
echo "[2/7] Generate/resume LTX train predictions"
"$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_gate \
  --manifest "$MATERIALIZED_MANIFEST" \
  --out_dir "$LTX_OUT" \
  --ltx_repo /workspace/third_party/LTX-Video \
  --ltx_python "$ROOT/ltxv_env/bin/python" \
  --pipeline_config /workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml \
  --view_type bdd \
  --limit "$TRAIN_LIMIT" \
  --prompt_mode legacy \
  --resume \
  --skip_bad

LTX_ROWS=$(( $(wc -l < "$LTX_OUT/report.csv") - 1 ))
if (( LTX_ROWS != TRAIN_LIMIT )); then
  echo "Expected $TRAIN_LIMIT completed LTX rows, found $LTX_ROWS. Stop before training." >&2
  exit 3
fi

echo
echo "[3/7] Materialize dense 40-frame train targets"
if [[ ! -s "$DENSE_MANIFEST" ]]; then
  "$PYTHON" -m wan_ticft.data.materialize_future_manifest \
    --src_jsonl "$LTX_OUT/selected_manifest.jsonl" \
    --out_jsonl "$DENSE_MANIFEST" \
    --frames_root "$DENSE_FRAMES" \
    --future_frames 40
else
  echo "resume dense manifest: $DENSE_MANIFEST"
fi

echo
echo "[4/7] Build transition16 and out-of-fold cnn_ltxmix65 train base"
"$PYTHON" -m wan_ticft.evaluation.build_ltx_condition_transitions \
  --manifest "$DENSE_MANIFEST" \
  --ltx_report "$LTX_OUT/report.csv" \
  --out_root "$TRANSITION_ROOT" \
  --transitions 16 \
  --clean

"$PYTHON" -m wan_ticft.evaluation.materialize_cnn_ltx_region_selector_variants \
  --manifest "$DENSE_MANIFEST" \
  --ltx_report "$TRANSITION_REPORT" \
  --out_root "$MATCHED_BASE_ROOT" \
  --epochs 8 \
  --batch_size 8 \
  --variant cnn_ltxmix65:ltx_mix:0.65 \
  --clean

BASE_ROWS=$(( $(wc -l < "$MATCHED_BASE_REPORT") - 1 ))
if (( BASE_ROWS != TRAIN_LIMIT )); then
  echo "Expected $TRAIN_LIMIT matched-base rows, found $BASE_ROWS. Stop before training." >&2
  exit 3
fi

echo
echo "[5/7] Train matched-base temporal feature refiner"
"$PYTHON" -m wan_ticft.evaluation.train_temporal_latent_refiner \
  --manifest "$DENSE_MANIFEST" \
  --base_report "$MATCHED_BASE_REPORT" \
  --train_manifest "$DENSE_MANIFEST" \
  --train_base_report "$MATCHED_BASE_REPORT" \
  --eval_manifest "$EVAL_MANIFEST" \
  --eval_base_report "$EVAL_BASE_REPORT" \
  --out_dir "$OUT" \
  --limit 24 \
  --train_limit "$TRAIN_LIMIT" \
  --eval_limit 24 \
  --train_rows "$TRAIN_LIMIT" \
  --epochs 10 \
  --batch_size 1 \
  --width 320 \
  --height 180 \
  --model_width 64 \
  --residual_scale 0.04 \
  --lr 0.0002 \
  --lambda_l1 1.0 \
  --lambda_ssim 0.25 \
  --lambda_residual 0.05 \
  --lambda_temporal 0.10 \
  --lambda_residual_temporal 0.20 \
  --lambda_image_feature 0.04 \
  --lambda_video_feature 0.04 \
  --feature_chunk_size 4 \
  --lossless_output \
  --clean

echo
echo "[6/7] Pixel and distribution metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_MANIFEST" \
  --reports \
    cnn_ltxmix65="$EVAL_BASE_REPORT" \
    temporal_matched16="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_MANIFEST" \
  --report cnn_ltxmix65="$EVAL_BASE_REPORT" \
  --report temporal_matched16="$OUT/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

echo
echo "[7/7] Comparison sheets"
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
  --reports \
    cnn_ltxmix65="$EVAL_BASE_REPORT" \
    temporal_matched16="$OUT/report.csv" \
  "${SHEET_ARGS[@]}" \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 220

echo
echo "Outputs:"
echo "  matched_base=$MATCHED_BASE_REPORT"
echo "  model=$OUT/temporal_latent_refiner.pt"
echo "  eval=${OUT}_eval.csv"
echo "  proxy=${OUT}_proxy.csv"
echo "  sheets=${OUT}_compare_sheets"
echo
echo "Gate rule:"
echo "  - Continue to matched60 only if PSNR improves >=0.10, SSIM does not fall,"
echo "    and neither FID nor FVD proxy worsens versus cnn_ltxmix65."
echo "  - Check sheets for reduced blur without geometry shift, halos, or flicker."
