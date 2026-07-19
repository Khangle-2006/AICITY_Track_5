#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/LTX-Video
OUT=/workspace/experiments/wan_ticft/ltxv_bdd_smoke4

echo "=== LTX-Video 2B extension feasibility audit ==="
echo "Driver:"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader
echo "Disk:"
df -h /workspace
echo "Python:"
/opt/conda/envs/wan22/bin/python --version

echo "=== Prepare four LTX-compatible BDD history clips ==="
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.prepare_ltxv_bdd_smoke \
  --index_dir /workspace/experiments/index \
  --out_dir "$OUT" \
  --limit 4 \
  --fps 15

echo "=== Clone official LTX-Video source only ==="
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/Lightricks/LTX-Video.git "$REPO"
fi

echo
echo "=== Feasibility gate result ==="
cat "$OUT/report.csv"
echo
echo "Repository: $REPO"
echo "No environment was installed and no checkpoint was downloaded."
echo "Next gate: run one 2B distilled video-extension sample only."
