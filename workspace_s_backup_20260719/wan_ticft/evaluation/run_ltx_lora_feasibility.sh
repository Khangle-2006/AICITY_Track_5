#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
REPO=/workspace/third_party/LTX-Video-Trainer
MANIFEST=/workspace/experiments/wan_ticft/manifests/train_ticft_balanced_121f_frames.jsonl

echo "=== Official LTX-Video Trainer feasibility audit ==="
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo "Disk:"
df -h /workspace
echo "Manifest: $MANIFEST"

echo
echo "=== Audit existing 500-sample forecasting dataset ==="
"$PYTHON" -m wan_ticft.evaluation.audit_ltx_lora_dataset \
  --manifest "$MANIFEST"

echo
echo "=== Clone official trainer source only ==="
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/Lightricks/LTX-Video-Trainer.git "$REPO"
fi
git -C "$REPO" log -1 --oneline

echo
echo "=== Relevant trainer configs/docs ==="
find "$REPO" -maxdepth 4 -type f \
  \( -iname '*2b*lora*' -o -iname '*lora*2b*' -o -iname '*low*vram*' -o -iname '*dataset*' \) \
  | sort \
  | head -40

echo
echo "=== Gate result ==="
echo "No environment was installed, no checkpoint was downloaded, and no training was launched."
echo "Dataset proposal: final condition frame + 120 future frames = 121-frame training clip."
echo "Next gate: inspect the cloned trainer format, then materialize a tiny 20-clip smoke dataset."
