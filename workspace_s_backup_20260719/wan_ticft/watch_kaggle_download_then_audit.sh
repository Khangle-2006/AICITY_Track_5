#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_PID="${1:-}"
if [[ -n "$DOWNLOAD_PID" ]]; then
  while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do
    sleep 60
  done
fi

echo "download process finished at $(date)"
echo "download log tail:"
tail -40 /data/kaggle/download_wts_video3.log || true

cd /workspace
export PYTHONPATH=/workspace
/opt/conda/envs/wan22/bin/python /workspace/wan_ticft/scan_test_overlap_visual.py \
  --root /data/kaggle/hungsvdut/wts-video-3 \
  --out_csv /workspace/experiments/wan_ticft/kaggle_wts_video3_test_overlap_visual_seek15.csv \
  --stride 15 \
  --topk 8 \
  --refine_radius 20 \
  --seek \
  > /data/kaggle/wts_video3_test_overlap_visual.log 2>&1
cp /workspace/experiments/wan_ticft/kaggle_wts_video3_test_overlap_visual_seek15.csv \
  /data/kaggle/wts_video3_test_overlap_visual_seek15.csv
echo "audit finished at $(date)"
