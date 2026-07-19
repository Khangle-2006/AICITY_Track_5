#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/sequence_retrieval_audit
LIMIT="${LIMIT:-8}"
REFERENCE_STRIDE="${REFERENCE_STRIDE:-12}"
TOP_VIDEOS="${TOP_VIDEOS:-8}"

echo "=== Track 5 sequence retrieval / duplicate audit ==="
echo "This is CPU-only and does not build a submission."
echo "out=$OUT"
echo "limit=$LIMIT reference_stride=$REFERENCE_STRIDE top_videos=$TOP_VIDEOS"
echo "The per-video index cache is resumable. The first run may take a long time."

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.audit_sequence_retrieval \
  --index_dir /workspace/experiments/index \
  --out_dir "$OUT" \
  --reference_splits train val \
  --reference_stride "$REFERENCE_STRIDE" \
  --top_videos "$TOP_VIDEOS" \
  --sequence_length 5 \
  --sequence_strides 1 2 3 4 \
  --limit "$LIMIT" \
  --sheet_top_n 3

echo
echo "Review:"
echo "  $OUT/summary.csv"
echo "  $OUT/matches.csv"
echo "  $OUT/sheets"
echo
echo "Strong duplicate evidence requires all of:"
echo "  - low sequence MAD across multiple history frames"
echo "  - high ORB homography inlier ratio"
echo "  - visually identical scene/object layout in the sheet"
echo "Do not extract future frames unless dataset rules permit it."
