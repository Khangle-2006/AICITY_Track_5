from __future__ import annotations

import argparse
import csv
import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def numeric_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)


def dhash_rgb(rgb: np.ndarray, size: int = 16) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def read_rgb(path: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mae_resized(a: np.ndarray, b: np.ndarray) -> float:
    if b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


@dataclass
class Query:
    sample_id: str
    source_folder: str
    view_type: str
    positions: list[int]
    frames: list[np.ndarray]
    first_hash: int


def load_queries(manifest: Path) -> list[Query]:
    queries: list[Query] = []
    for row in read_jsonl(manifest):
        paths = numeric_pngs(Path(row["input_dir"]))
        if not paths:
            continue
        raw_positions = sorted(set([0, len(paths) // 2, len(paths) - 1]))
        frames = []
        positions = []
        for pos in raw_positions:
            frame = read_rgb(paths[pos])
            if frame is not None:
                positions.append(pos)
                frames.append(frame)
        if not frames:
            continue
        queries.append(
            Query(
                sample_id=row["sample_id"],
                source_folder=row["source_folder"],
                view_type=row["view_type"],
                positions=positions,
                frames=frames,
                first_hash=dhash_rgb(frames[0]),
            )
        )
    return queries


def push_top(heap: list[tuple[int, str, int]], item: tuple[int, str, int], topk: int) -> None:
    # Store negative distance so heap[0] is the worst of the kept best matches.
    neg_item = (-item[0], item[1], item[2])
    if len(heap) < topk:
        heapq.heappush(heap, neg_item)
    elif neg_item > heap[0]:
        heapq.heapreplace(heap, neg_item)


def get_video_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def verify_sequence(query: Query, video_path: Path, start_frame: int, refine_radius: int) -> dict[str, str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"verify_status": "cannot_open"}

    best = None
    for offset in range(-refine_radius, refine_radius + 1):
        start = max(0, start_frame + offset)
        dists = []
        maes = []
        ok = True
        for pos, qframe in zip(query.positions, query.frames):
            frame = get_video_frame(cap, start + pos)
            if frame is None:
                ok = False
                break
            dists.append(hamming(dhash_rgb(frame), dhash_rgb(qframe)))
            maes.append(mae_resized(qframe, frame))
        if not ok:
            continue
        score = float(np.mean(maes)) + 0.25 * float(np.mean(dists))
        candidate = (score, start, dists, maes)
        if best is None or candidate < best:
            best = candidate
    cap.release()

    if best is None:
        return {"verify_status": "no_frames"}
    score, start, dists, maes = best
    return {
        "verify_status": "ok",
        "refined_start": str(start),
        "seq_hamming_mean": f"{float(np.mean(dists)):.3f}",
        "seq_hamming_max": str(int(max(dists))),
        "seq_mae_mean": f"{float(np.mean(maes)):.3f}",
        "seq_mae_max": f"{float(max(maes)):.3f}",
        "seq_score": f"{score:.3f}",
    }


def scan_videos(
    queries: list[Query],
    roots: list[Path],
    stride: int,
    topk: int,
    seek: bool,
) -> dict[str, list[tuple[int, str, int]]]:
    heaps: dict[str, list[tuple[int, str, int]]] = {query.sample_id: [] for query in queries}
    videos = []
    for root in roots:
        if root.exists():
            videos.extend(sorted(root.rglob("*.mp4")))
    print(f"videos={len(videos)} stride={stride} queries={len(queries)}", flush=True)

    for video_index, video in enumerate(videos, start=1):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        scanned = 0
        if seek:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            positions = range(0, max(1, frame_count), stride) if frame_count > 0 else range(0, 1)
            for frame_index in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frame_hash = dhash_rgb(rgb)
                for query in queries:
                    dist = hamming(query.first_hash, frame_hash)
                    push_top(heaps[query.sample_id], (dist, str(video), frame_index), topk)
                scanned += 1
        else:
            frame_index = 0
            while True:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                if frame_index % stride == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frame_hash = dhash_rgb(rgb)
                    for query in queries:
                        dist = hamming(query.first_hash, frame_hash)
                        push_top(heaps[query.sample_id], (dist, str(video), frame_index), topk)
                    scanned += 1
                frame_index += 1
        cap.release()
        if video_index % 100 == 0:
            print(f"[{video_index}/{len(videos)}] scanned_frames={scanned} last={video}", flush=True)

    out: dict[str, list[tuple[int, str, int]]] = {}
    for query in queries:
        items = [(-dist, path, frame) for dist, path, frame in heaps[query.sample_id]]
        out[query.sample_id] = sorted(items, key=lambda item: item[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/workspace/experiments/wan_ticft/manifests/test_ticft_frames.jsonl"))
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--refine_radius", type=int, default=8)
    parser.add_argument("--seek", action="store_true")
    args = parser.parse_args()

    queries = load_queries(args.manifest)
    matches = scan_videos(queries, args.root, args.stride, args.topk, args.seek)
    query_by_id = {query.sample_id: query for query in queries}

    rows = []
    for query in queries:
        for rank, (dist, video_path, frame_index) in enumerate(matches[query.sample_id], start=1):
            verify = verify_sequence(query, Path(video_path), frame_index, args.refine_radius)
            rows.append(
                {
                    "sample_id": query.sample_id,
                    "source_folder": query.source_folder,
                    "view_type": query.view_type,
                    "rank": rank,
                    "first_frame_hamming": dist,
                    "coarse_start": frame_index,
                    "video_path": video_path,
                    **verify,
                }
            )
        if not matches[query.sample_id]:
            rows.append(
                {
                    "sample_id": query.sample_id,
                    "source_folder": query.source_folder,
                    "view_type": query.view_type,
                    "rank": "",
                    "first_frame_hamming": "",
                    "coarse_start": "",
                    "video_path": "",
                    "verify_status": "no_match",
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "source_folder",
        "view_type",
        "rank",
        "first_frame_hamming",
        "coarse_start",
        "refined_start",
        "seq_hamming_mean",
        "seq_hamming_max",
        "seq_mae_mean",
        "seq_mae_max",
        "seq_score",
        "video_path",
        "verify_status",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    strong = [
        row for row in rows
        if row.get("verify_status") == "ok"
        and row.get("seq_mae_mean")
        and float(row["seq_mae_mean"]) <= 6.0
        and float(row["seq_hamming_mean"]) <= 12.0
    ]
    print(f"out={args.out_csv}", flush=True)
    print(f"rows={len(rows)} strong_visual_duplicates={len(strong)}", flush=True)
    for row in strong[:20]:
        print(row, flush=True)


if __name__ == "__main__":
    main()
