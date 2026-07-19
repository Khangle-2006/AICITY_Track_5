from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("KAGGLEHUB_CACHE", "/data/kagglehub")

import kagglehub  # noqa: E402


handle = "hungsvdut/wts-video-3"
target = Path("/data/kaggle/hungsvdut/wts-video-3")
target.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading {handle} with KAGGLEHUB_CACHE={os.environ['KAGGLEHUB_CACHE']}", flush=True)
path = Path(kagglehub.dataset_download(handle))
print(f"cache_path={path}", flush=True)

if target.exists() or target.is_symlink():
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.resolve() != path.resolve():
        shutil.rmtree(target)

if not target.exists():
    try:
        target.symlink_to(path, target_is_directory=True)
        print(f"symlink={target} -> {path}", flush=True)
    except OSError:
        print(f"copying to {target}", flush=True)
        shutil.copytree(path, target)

print("top files:", flush=True)
for item in list(target.rglob("*"))[:50]:
    print(item, flush=True)
