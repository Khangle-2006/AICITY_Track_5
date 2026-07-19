#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl}
LTX_REPO=${LTX_REPO:-/workspace/third_party/LTX-Video}
LTX_PYTHON=${LTX_PYTHON:-$ROOT/ltxv_env/bin/python}
PIPELINE_CONFIG=${PIPELINE_CONFIG:-/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml}
DEPTH_REPORT=${DEPTH_REPORT:-$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv}
PROMPT_MODES="${PROMPT_MODES:-traffic_detail copy_bias dashcam_native minimal}"
CHUNK_FRAMES="${CHUNK_FRAMES:-16}"
CUDA_ID="${CUDA_VISIBLE_DEVICES:-2}"
CUDA_ID="${CUDA_ID%%,*}"

SAMPLES=(
  val_bdd_pc_5k_video7040_phase0
  val_bdd_pc_5k_video8139_phase2
  val_bdd_pc_5k_video8170_phase4
  val_bdd_pc_5k_video3237_phase0
)

export CPU_THREADS="${CPU_THREADS:-4}"
export OMP_NUM_THREADS=$CPU_THREADS OPENBLAS_NUM_THREADS=$CPU_THREADS MKL_NUM_THREADS=$CPU_THREADS NUMEXPR_NUM_THREADS=$CPU_THREADS VECLIB_MAXIMUM_THREADS=$CPU_THREADS
export CUDA_VISIBLE_DEVICES="$CUDA_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== LTX2B chunked prompt smoke4 ==="
echo "prompt_modes=$PROMPT_MODES"
echo "chunk_frames=$CHUNK_FRAMES cuda=$CUDA_ID"
echo "samples=${SAMPLES[*]}"

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$CUDA_ID" 12000

for mode in $PROMPT_MODES; do
  out="$ROOT/ltx2b_chunked_prompt_smoke4_${mode}_c${CHUNK_FRAMES}"
  echo
  echo ">>> PROMPT_MODE=$mode OUT=$out"
  "$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_chunked_gate \
    --manifest "$MANIFEST" \
    --out_dir "$out" \
    --ltx_repo "$LTX_REPO" \
    --ltx_python "$LTX_PYTHON" \
    --pipeline_config "$PIPELINE_CONFIG" \
    --limit "${#SAMPLES[@]}" \
    --view_type bdd \
    --sample_ids "${SAMPLES[@]}" \
    --prompt_mode "$mode" \
    --height 704 \
    --width 1216 \
    --chunk_frames "$CHUNK_FRAMES" \
    --skip_bad

  "$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
    --manifest "$out/selected_manifest.jsonl" \
    --reports ltx2b_chunked="$out/report.csv" depth_a75="$DEPTH_REPORT" \
    --include_repeat_last \
    --offset_mode timestamp \
    --out_csv "$out/eval_pixel.csv"

  for sample_id in "${SAMPLES[@]}"; do
    "$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
      --manifest "$out/selected_manifest.jsonl" \
      --reports ltx2b_chunked="$out/report.csv" depth_a75="$DEPTH_REPORT" \
      --sample_id "$sample_id" \
      --out_dir "$out/compare_sheets" \
      --frames 8 \
      --tile_width 220 || true
  done
done

"$PYTHON" - <<'PY'
import csv
from pathlib import Path
root = Path("/workspace/experiments/wan_ticft")
modes = "traffic_detail copy_bias dashcam_native minimal".split()
print(f"{'mode':<16} {'N':>3} {'PSNR':>8} {'SSIM':>8} {'AbsDiff':>8}")
print("-" * 50)
for mode in modes:
    path = root / f"ltx2b_chunked_prompt_smoke4_{mode}_c16" / "eval_pixel.csv"
    rows = [r for r in csv.DictReader(path.open()) if r.get("candidate") == "ltx2b_chunked" and r.get("status") == "ok"] if path.exists() else []
    if not rows:
        print(f"{mode:<16} missing")
        continue
    avg = lambda key: sum(float(r[key]) for r in rows) / len(rows)
    print(f"{mode:<16} {len(rows):>3} {avg('psnr'):>8.4f} {avg('ssim'):>8.4f} {avg('absdiff'):>8.4f}")
PY
