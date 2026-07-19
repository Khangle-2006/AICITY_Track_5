from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


MANIFEST = Path("/workspace/experiments/wan_ticft/manifests/test_ticft_frames.jsonl")
KAGGLE_ROOT = Path("/data/kaggle/hungsvdut/wts-video-3")
OUT = Path("/workspace/experiments/wan_ticft/kaggle_wts_video3_test_overlap_audit.csv")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def numeric_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)


def dhash_rgb(rgb: np.ndarray, size: int = 16) -> str:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:0{size * size // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def read_png_hash(path: Path) -> str:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return ""
    return dhash_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def read_video_hash(video: Path, index: int = 0) -> str:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return ""
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return ""
    return dhash_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def main() -> None:
    if not KAGGLE_ROOT.exists():
        raise SystemExit(f"missing {KAGGLE_ROOT}")
    test_rows = read_jsonl(MANIFEST)
    videos = sorted(KAGGLE_ROOT.rglob("*.mp4"))
    video_by_stem: dict[str, list[Path]] = {}
    for video in videos:
        video_by_stem.setdefault(video.stem, []).append(video)

    test_hashes = {}
    for row in test_rows:
        pngs = numeric_pngs(Path(row["input_dir"]))
        test_hashes[row["source_folder"]] = read_png_hash(pngs[0]) if pngs else ""

    # Full visual first-frame scan is cheap enough for this dataset and catches renamed overlaps.
    video_hashes = []
    for i, video in enumerate(videos, start=1):
        h = read_video_hash(video, 0)
        video_hashes.append((video, h))
        if i % 500 == 0:
            print(f"hashed {i}/{len(videos)}", flush=True)

    report = []
    for row in test_rows:
        source = row["source_folder"]
        query_hash = test_hashes.get(source, "")
        name_matches = video_by_stem.get(source, [])
        near = []
        if query_hash:
            for video, h in video_hashes:
                if not h:
                    continue
                dist = hamming_hex(query_hash, h)
                if dist <= 8:
                    near.append((dist, video))
            near.sort(key=lambda x: x[0])

        if name_matches or near:
            candidates = [(None, p, "name") for p in name_matches] + [(d, p, "visual") for d, p in near[:10]]
            for dist, path, kind in candidates:
                report.append(
                    {
                        "sample_id": row["sample_id"],
                        "source_folder": source,
                        "view_type": row["view_type"],
                        "match_kind": kind,
                        "hamming": "" if dist is None else dist,
                        "kaggle_video": str(path),
                    }
                )
        else:
            report.append(
                {
                    "sample_id": row["sample_id"],
                    "source_folder": source,
                    "view_type": row["view_type"],
                    "match_kind": "none",
                    "hamming": "",
                    "kaggle_video": "",
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        fields = ["sample_id", "source_folder", "view_type", "match_kind", "hamming", "kaggle_video"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report)
    print(f"kaggle_videos={len(videos)}")
    print(f"rows={len(report)}")
    print(f"name_matches={sum(1 for r in report if r['match_kind'] == 'name')}")
    print(f"visual_matches={sum(1 for r in report if r['match_kind'] == 'visual')}")
    print(f"out={OUT}")


if __name__ == "__main__":
    main()
