#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/ati_bdd_smoke4
ATI=/workspace/third_party/ATI

echo "=== Resource audit ==="
df -h /workspace
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

echo "=== Prepare four BDD history clips ==="
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.prepare_ati_bdd_smoke \
  --index_dir /workspace/experiments/index \
  --out_dir "$OUT" \
  --limit 4

echo "=== Prepare ATI repository ==="
if [[ ! -d "$ATI/.git" ]]; then
  git clone --depth 1 https://github.com/bytedance/ATI.git "$ATI"
fi

echo "=== Feasibility result ==="
echo "Prepared history clips: $OUT/history_videos"
echo "ATI repo: $ATI"
echo
echo "Next gated step:"
echo "  1. Install ATI in a dedicated environment."
echo "  2. Run ATI motion extraction on these four clips."
echo "  3. Inspect trajectories before downloading/running the 14B model."

