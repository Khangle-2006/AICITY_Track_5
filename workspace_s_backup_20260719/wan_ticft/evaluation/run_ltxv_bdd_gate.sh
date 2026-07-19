#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/LTX-Video
ENV=/workspace/experiments/wan_ticft/ltxv_env
IN=/workspace/experiments/wan_ticft/ltxv_bdd_smoke4/history_videos/video1170.mp4
REPORT=/workspace/experiments/wan_ticft/ltxv_bdd_smoke4/report.csv
OUT=/workspace/experiments/wan_ticft/ltxv_bdd_gate_video1170
CONFIG=/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== LTX-Video 2B distilled extension gate: video1170 ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 24000

if [[ ! -x "$ENV/bin/python" ]]; then
  /opt/conda/envs/wan22/bin/python -m venv --system-site-packages "$ENV"
fi

"$ENV/bin/python" -m pip install -e "$REPO[inference]"

PROMPT="$("$ENV/bin/python" - "$REPORT" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    row = next(csv.DictReader(f))
print(
    "Continue the observed forward-facing traffic video without changing the road, "
    "weather, lighting, camera viewpoint, or existing object identities. "
    + row["caption_pedestrian"]
    + " "
    + row["caption_vehicle"]
    + " Maintain physically plausible traffic motion, sharp geometry, and temporal consistency."
)
PY
)"

rm -rf "$OUT"
mkdir -p "$OUT/videos"

HF_HUB_DOWNLOAD_TIMEOUT=3600 TOKENIZERS_PARALLELISM=false \
PYTHONPATH="$REPO" \
"$ENV/bin/python" "$REPO/inference.py" \
  --prompt "$PROMPT" \
  --conditioning_media_paths "$IN" \
  --conditioning_start_frames 0 \
  --conditioning_strengths 1.0 \
  --height 704 \
  --width 1216 \
  --num_frames 121 \
  --frame_rate 15 \
  --seed 1 \
  --offload_to_cpu true \
  --pipeline_config "$CONFIG" \
  --output_path "$OUT/videos"

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$OUT/videos" \
  --mode videos \
  --out_dir "${OUT}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

echo "Inspect: ${OUT}_sheets"
