#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/vdpm_bdd_smoke4
REPO=/workspace/third_party/vdpm

echo "=== V-DPM geometry-residual feasibility audit ==="
df -h /workspace
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

echo "=== Prepare complementary BDD histories ==="
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.prepare_vdpm_bdd_smoke \
  --ceiling_csv /workspace/experiments/wan_ticft/ltx_region_composite_ceiling.csv \
  --history_dir /workspace/experiments/wan_ticft/ltxv_bdd25_gate/history_videos \
  --out_dir "$OUT"

echo "=== Clone V-DPM source only ==="
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/eldar/vdpm.git "$REPO"
fi

echo "=== Relevant V-DPM files ==="
find "$REPO" -maxdepth 2 -type f \
  \( -iname 'README*' -o -iname '*demo*' -o -iname '*infer*' -o -iname '*requirement*' \) \
  -print | sort

echo
echo "=== Gate result ==="
echo "Inputs: $OUT/history_videos"
echo "Selection report: $OUT/report.csv"
echo "Repository: $REPO"
echo "No environment was installed and no checkpoint was downloaded."
echo "Next: audit the official inference contract, then reconstruct these four histories."
