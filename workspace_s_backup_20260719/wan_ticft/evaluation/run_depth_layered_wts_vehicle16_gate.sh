#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
BASE_ROOT="$ROOT/depth_layered_wts_vehicle16_base"
BASE_REPORT="$BASE_ROOT/region_norm_medium_report.csv"
OUT="$ROOT/depth_layered_wts_vehicle16"
SELECTED_MANIFEST="$OUT/selected_manifest.jsonl"
MODEL_CACHE="$ROOT/depth_layered_models"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Depth-aware renderer: WTS vehicle/IP 16-clip scale gate ==="
[[ -f "$MANIFEST" ]] || { echo "Missing input: $MANIFEST" >&2; exit 2; }
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 12000

echo "[1/4] Materialize 16 vehicle/IP validation bases"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$MANIFEST" \
  --out_dir "$BASE_ROOT" \
  --views wts_vehicle_or_ip_camera \
  --limit_per_domain 16 \
  --cover_manifest_targets \
  --candidates region_norm_medium \
  --clean

echo "[2/4] Render control, fixed-alpha, and horizon-decayed variants"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME="$MODEL_CACHE/hf_home" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_root "$OUT" \
  --model_cache "$MODEL_CACHE/huggingface" \
  --view_type wts_vehicle_or_ip_camera \
  --preserve_full_video \
  --limit 16 \
  --alphas 0.00 0.75 \
  --horizon_decay \
  --horizon_decay_start 40 \
  --horizon_decay_end 80 \
  --horizon_decay_alpha 0.75 \
  --horizon_decay_floor 0.25 \
  --focal_scale 0.9 \
  --point_stride 2 \
  --splat_radius 1 \
  --min_matches 20 \
  --min_inliers 12 \
  --photo_sigma 32 \
  --segment_batch 4 \
  --pose_smoothing_window 7 \
  --filled_pose_scale 0.35 \
  --confidence_ema 0.65 \
  --clean

echo "[preflight] Verify every target offset is readable"
"$PYTHON" - <<PY
import json
from pathlib import Path
from baseline.frame_io import import_cv2_np
from wan_ticft.evaluation.evaluate_video_reports import map_target_offsets, read_report, read_video_rgb_frames, video_fps

cv2, _ = import_cv2_np()
rows = [json.loads(line) for line in open("$SELECTED_MANIFEST", encoding="utf-8") if line.strip()]
paths = {
    "lossless_control": Path("$OUT/depth_layered_a00/report.csv"),
    "depth_a75": Path("$OUT/depth_layered_a75/report.csv"),
    "depth_hdecay": Path("$OUT/depth_layered_hdecay/report.csv"),
}
reports = {name: read_report(path) for name, path in paths.items()}
failures = []
for row in rows:
    for name, report in reports.items():
        video = report.get(row["sample_id"])
        offsets = map_target_offsets(row, video_fps(cv2, video))
        found = read_video_rgb_frames(cv2, video, offsets)
        missing = [index for index in offsets if index not in found]
        if missing:
            failures.append(f"{row['sample_id']} {name}: missing offsets {missing}")
if failures:
    raise RuntimeError("preflight failed:\n" + "\n".join(failures))
print(f"preflight ok: {len(rows)} samples x {len(reports)} reports")
PY

echo "[3/4] Pixel and distribution gate"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$SELECTED_MANIFEST" \
  --reports \
    region_norm_medium="$BASE_REPORT" \
    lossless_control="$OUT/depth_layered_a00/report.csv" \
    depth_a75="$OUT/depth_layered_a75/report.csv" \
    depth_hdecay="$OUT/depth_layered_hdecay/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$SELECTED_MANIFEST" \
  --report lossless_control="$OUT/depth_layered_a00/report.csv" \
  --report depth_a75="$OUT/depth_layered_a75/report.csv" \
  --report depth_hdecay="$OUT/depth_layered_hdecay/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 8 \
  --frame_stride 1

echo "[4/4] Contact sheets"
mapfile -t SAMPLE_IDS < <("$PYTHON" - <<PY
import json
for line in open("$SELECTED_MANIFEST", encoding="utf-8"):
    if line.strip():
        print(json.loads(line)["sample_id"])
PY
)
SHEET_ARGS=()
for sample_id in "${SAMPLE_IDS[@]}"; do
  SHEET_ARGS+=(--sample_id "$sample_id")
done
"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$SELECTED_MANIFEST" \
  --reports \
    lossless_control="$OUT/depth_layered_a00/report.csv" \
    depth_a75="$OUT/depth_layered_a75/report.csv" \
    depth_hdecay="$OUT/depth_layered_hdecay/report.csv" \
  "${SHEET_ARGS[@]}" \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 220

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "diagnostics=$OUT/diagnostics.csv"
echo "sheets=${OUT}_compare_sheets"
echo "Gate: promote only with broad pixel gains, improved FID/FVD, and clean sheets."
