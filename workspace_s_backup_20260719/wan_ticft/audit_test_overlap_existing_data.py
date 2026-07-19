from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


MANIFEST = Path("/workspace/experiments/wan_ticft/manifests/test_ticft_frames.jsonl")
EXTERNAL_VIDEO = Path("/data/Track_5/WTS_dataset/WTS_dataset/external/video")
WTS_VIDEO = Path("/data/Track_5/WTS_dataset/WTS_dataset/video")
OUT = Path("/workspace/experiments/wan_ticft/test_overlap_existing_data_audit.csv")


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


def image_sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def read_video_frame(video: Path, index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_video_map(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if not root.exists():
        return out
    for video in root.rglob("*.mp4"):
        out.setdefault(video.stem, []).append(video)
    return out


def main() -> None:
    rows = read_jsonl(MANIFEST)
    external = build_video_map(EXTERNAL_VIDEO)
    wts = build_video_map(WTS_VIDEO)
    report = []

    for row in rows:
        source = row["source_folder"]
        input_dir = Path(row["input_dir"])
        input_paths = numeric_pngs(input_dir)
        first_sha = image_sha(input_paths[0]) if input_paths else ""
        first_hash = ""
        last_hash = ""
        if input_paths:
            first = cv2.imread(str(input_paths[0]), cv2.IMREAD_COLOR)
            last = cv2.imread(str(input_paths[-1]), cv2.IMREAD_COLOR)
            if first is not None:
                first_hash = dhash_rgb(cv2.cvtColor(first, cv2.COLOR_BGR2RGB))
            if last is not None:
                last_hash = dhash_rgb(cv2.cvtColor(last, cv2.COLOR_BGR2RGB))

        candidates = []
        for origin, mapping in [("external", external), ("wts_trainval", wts)]:
            for video in mapping.get(source, []):
                candidates.append((origin, video))

        if not candidates:
            report.append(
                {
                    "sample_id": row["sample_id"],
                    "source_folder": source,
                    "view_type": row["view_type"],
                    "name_match": "no",
                    "match_origin": "",
                    "match_video": "",
                    "input_frames": len(input_paths),
                    "frame0_hamming": "",
                    "last_hamming": "",
                    "exact_frame0_sha": first_sha,
                    "notes": "",
                }
            )
            continue

        for origin, video in candidates:
            vf0 = read_video_frame(video, 0)
            vlast = read_video_frame(video, max(0, len(input_paths) - 1))
            h0 = hamming_hex(first_hash, dhash_rgb(vf0)) if first_hash and vf0 is not None else ""
            hl = hamming_hex(last_hash, dhash_rgb(vlast)) if last_hash and vlast is not None else ""
            report.append(
                {
                    "sample_id": row["sample_id"],
                    "source_folder": source,
                    "view_type": row["view_type"],
                    "name_match": "yes",
                    "match_origin": origin,
                    "match_video": str(video),
                    "input_frames": len(input_paths),
                    "frame0_hamming": h0,
                    "last_hamming": hl,
                    "exact_frame0_sha": first_sha,
                    "notes": "visual_near_match" if isinstance(h0, int) and h0 <= 8 else "",
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "sample_id",
            "source_folder",
            "view_type",
            "name_match",
            "match_origin",
            "match_video",
            "input_frames",
            "frame0_hamming",
            "last_hamming",
            "exact_frame0_sha",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report)

    print(f"rows={len(report)} out={OUT}")
    print(f"name_matches={sum(1 for r in report if r['name_match'] == 'yes')}")
    print(f"near_matches={sum(1 for r in report if r['notes'] == 'visual_near_match')}")
    for r in report:
        if r["name_match"] == "yes":
            print(r)


if __name__ == "__main__":
    main()
