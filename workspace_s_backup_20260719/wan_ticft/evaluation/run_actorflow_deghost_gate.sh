#!/bin/bash
set -euo pipefail
# Gate actor-flow deghost over an existing mf_video validation render.
cd /workspace
export CPU_THREADS="${CPU_THREADS:-4}"
export OMP_NUM_THREADS=$CPU_THREADS OPENBLAS_NUM_THREADS=$CPU_THREADS MKL_NUM_THREADS=$CPU_THREADS NUMEXPR_NUM_THREADS=$CPU_THREADS VECLIB_MAXIMUM_THREADS=$CPU_THREADS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=/workspace/experiments/wan_ticft
MANIFEST=$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl
MODEL_CACHE=$ROOT/depth_layered_models/huggingface
REAL=$ROOT/frames_balanced_val_125/val
A75DIR=$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75
BASE=${BASE:-$ROOT/basemode_gate/mf_a75_h4_video}
OUT=$ROOT/actorflow_gate

if [[ ! -f "$BASE/report.csv" ]]; then
  echo "Missing base report: $BASE/report.csv"
  echo "Run run_basemode_gate.sh first, or set BASE=/path/to/mf_video_candidate"
  exit 2
fi

rm -rf "$OUT"
echo "=== render actor-flow deghost ==="
"$PYTHON" -m wan_ticft.evaluation.materialize_actor_flow_deghost \
  --manifest "$MANIFEST" --base_report "$BASE/report.csv" --out_root "$OUT" \
  --model_cache "$MODEL_CACHE" --view_type bdd --limit 24 --tag mf_actorflow_h4 \
  --flow_scale "${FLOW_SCALE:-0.5}" --actor_strength "${ACTOR_STRENGTH:-0.85}" \
  --mask_dilate "${MASK_DILATE:-17}" --mask_blur "${MASK_BLUR:-5.0}" \
  --segment_batch 4

echo "=== PIXEL vs GT ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports --manifest "$MANIFEST" \
  --reports a75_single="$A75DIR/report.csv" mf_video="$BASE/report.csv" mf_actorflow="$OUT/mf_actorflow_h4/report.csv" \
  --offset_mode timestamp --out_csv "$OUT/eval_pixel.csv"

echo "=== REAL-FID vs GT ==="
"$PYTHON" -m wan_ticft.evaluation.compute_real_fid --real_root "$REAL" --work "$OUT/_fid" --limit 24 \
  --candidate "a75_single=$A75DIR/videos" \
  --candidate "mf_video=$BASE/videos" \
  --candidate "mf_actorflow=$OUT/mf_actorflow_h4/videos"

echo "=== Win = mf_actorflow improves REAL-FID/FVD proxy or visual ghosting without pixel collapse ==="
