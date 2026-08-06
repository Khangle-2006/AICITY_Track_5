#!/usr/bin/env bash
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$SOLUTION_DIR:${PYTHONPATH:-}"

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate}
ZIP_PATH=${ZIP_PATH:-${SUBMISSION}.zip}

echo "=== Verify and zip Track 5 submission ==="
echo "index=$INDEX"
echo "submission=$SUBMISSION"
echo "zip_path=$ZIP_PATH"

"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$SUBMISSION" \
  --submission_dir "$SUBMISSION" \
  --verify_only

rm -f "$ZIP_PATH"
"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$SUBMISSION" \
  --submission_dir "$SUBMISSION" \
  --zip "$ZIP_PATH"

echo
echo "Created:"
echo "  $ZIP_PATH"
