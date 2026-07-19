#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/experiments/wan_ticft/wan22_wts100_s10
SOURCE="$ROOT/shard_00_gpu3/videos"
DEST="$ROOT/shard_00_gpu0/videos"

mkdir -p "$DEST"

linked=0
existing=0
for source in "$SOURCE"/*.mp4; do
  [[ -f "$source" ]] || continue
  destination="$DEST/$(basename "$source")"
  if [[ -e "$destination" ]]; then
    existing=$((existing + 1))
    continue
  fi
  ln -s "$source" "$destination"
  linked=$((linked + 1))
done

total="$(find "$DEST" -maxdepth 1 -type f,l -name '*.mp4' | wc -l)"
echo "linked: $linked"
echo "already existed: $existing"
echo "GPU 0 shard visible videos: $total / 100"
