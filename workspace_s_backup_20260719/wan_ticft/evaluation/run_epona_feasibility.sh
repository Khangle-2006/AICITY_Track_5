#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/Epona

echo "=== Epona source-only feasibility audit ==="

if [[ ! -d "$REPO/.git" ]]; then
  if [[ -e "$REPO" ]]; then
    echo "FAIL: $REPO exists but is not a git checkout; inspect it manually."
    exit 2
  fi
  git clone --depth 1 https://github.com/Kevin-thu/Epona.git "$REPO"
fi

echo
echo "Commit:"
git -C "$REPO" rev-parse --short HEAD

echo
echo "Custom/self-predicted inference references:"
grep -RInE \
  'test_demo|self.pred|predict.*traj|pose.npy|yaw.npy|condition_frames|num_frames|video_length' \
  "$REPO/scripts/test" "$REPO/configs" "$REPO/dataset" "$REPO/data_preparation" \
  | head -n 240

echo
echo "Checkpoint/config references:"
grep -RInE \
  'resume_path|vae_ckpt|resolution|image_size|condition_frames|video_length' \
  "$REPO/scripts/test/test_demo.py" "$REPO/configs" \
  | head -n 240

echo
echo "Environment references:"
grep -nE 'torch|torchvision|flash|xformers|deepspeed' "$REPO/requirements.txt" | head -n 160

echo
echo "Audit rule:"
echo "  - Pass only if test_demo accepts custom RGB history and predicts future"
echo "    trajectory/video without requiring ground-truth future pose or NuPlan metadata."
echo "  - Confirm condition-frame count, output resolution/horizon, checkpoint size,"
echo "    and single-GPU compatibility before any download or environment install."
echo "  - Stop if integration requires dataset-specific future actions or major source repair."
