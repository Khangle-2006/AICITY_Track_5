from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KAGGLEHUB_CACHE", "/data/kagglehub_probe")

import kagglehub  # noqa: E402


handle = "hungsvdut/wts-video-3"
out = Path("/data/kaggle/probe_wts_video3")
out.mkdir(parents=True, exist_ok=True)

names = ["video3806", "video1170", "video135", "video8958"]
patterns = []
for name in names:
    patterns.extend(
        [
            f"{name}.mp4",
            f"video/train/{name}.mp4",
            f"video/val/{name}.mp4",
            f"external/video/train/{name}.mp4",
            f"external/video/val/{name}.mp4",
            f"caption/train/{name}_caption.json",
            f"caption/val/{name}_caption.json",
            f"external/caption/train/{name}_caption.json",
            f"external/caption/val/{name}_caption.json",
        ]
    )

for path in patterns:
    try:
        got = kagglehub.dataset_download(handle, path=path, output_dir=str(out))
        print("FOUND", path, "->", got, flush=True)
    except Exception as exc:
        print("MISS", path, type(exc).__name__, str(exc)[:180].replace("\n", " "), flush=True)
