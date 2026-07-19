"""Audit whether test histories overlap train/validation source videos.

This is a provenance diagnostic, not a submission builder. It creates a
resumable coarse dHash index, refines the best candidates with multi-frame
pixel comparisons, and verifies geometry with ORB homography inliers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ORIENTATIONS = {
    "none": lambda image: image,
    "cw": lambda image: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
    "ccw": lambda image: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    "180": lambda image: cv2.rotate(image, cv2.ROTATE_180),
}
POPCOUNT = np.array([int(value).bit_count() for value in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class ReferenceVideo:
    video_id: int
    split: str
    dataset_part: str
    view_type: str
    source_folder: str
    video_path: Path


def numeric_images(directory: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        match = re.search(r"(\d+)(?!.*\d)", path.stem)
        return (int(match.group(1)) if match else -1, path.name)

    paths = [path for path in directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    return sorted(paths, key=key)


def dhash(image: np.ndarray) -> np.uint64:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    packed = np.packbits(bits.astype(np.uint8), bitorder="little")
    return packed.view(np.uint64)[0]


def hamming_many(values: np.ndarray, query: np.uint64) -> np.ndarray:
    xor = np.ascontiguousarray(values ^ query)
    return POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def canonical(image: np.ndarray, orientation: str, width: int = 256, height: int = 144) -> np.ndarray:
    image = ORIENTATIONS[orientation](image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def normalized_mad(left: np.ndarray, right: np.ndarray) -> float:
    left_f = left.astype(np.float32)
    right_f = right.astype(np.float32)
    left_f = (left_f - left_f.mean()) / max(float(left_f.std()), 1.0)
    right_f = (right_f - right_f.mean()) / max(float(right_f.std()), 1.0)
    return float(np.mean(np.abs(left_f - right_f)))


def pixel_mad(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(left, right)))


def orb_inlier_ratio(left: np.ndarray, right: np.ndarray) -> tuple[int, float]:
    orb = cv2.ORB_create(nfeatures=1200, fastThreshold=10)
    keypoints_left, descriptors_left = orb.detectAndCompute(left, None)
    keypoints_right, descriptors_right = orb.detectAndCompute(right, None)
    if descriptors_left is None or descriptors_right is None:
        return 0, 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors_left, descriptors_right, k=2)
    good = [first for first, second in matches if first.distance < 0.75 * second.distance]
    if len(good) < 8:
        return len(good), 0.0
    points_left = np.float32([keypoints_left[item.queryIdx].pt for item in good])
    points_right = np.float32([keypoints_right[item.trainIdx].pt for item in good])
    _, mask = cv2.findHomography(points_left, points_right, cv2.RANSAC, 3.0)
    return len(good), float(mask.mean()) if mask is not None else 0.0


def read_rows(index_dir: Path, split: str) -> list[dict[str, str]]:
    with (index_dir / f"{split}_index.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_reference_videos(index_dir: Path, splits: list[str]) -> list[ReferenceVideo]:
    unique: dict[str, dict[str, str]] = {}
    for split in splits:
        for row in read_rows(index_dir, split):
            raw_path = row.get("video_path", "")
            if raw_path and Path(raw_path).is_file():
                unique.setdefault(str(Path(raw_path).resolve()), row)
    videos = []
    for video_id, (raw_path, row) in enumerate(sorted(unique.items())):
        videos.append(
            ReferenceVideo(
                video_id=video_id,
                split=row["split"],
                dataset_part=row["dataset_part"],
                view_type=row["view_type"],
                source_folder=row["source_folder"],
                video_path=Path(raw_path),
            )
        )
    return videos


def shard_path(cache_dir: Path, video: ReferenceVideo, stride: int) -> Path:
    digest = hashlib.sha1(f"{video.video_path}|{stride}".encode()).hexdigest()[:16]
    return cache_dir / f"{video.video_id:05d}_{digest}.npz"


def index_video(video: ReferenceVideo, cache_dir: Path, stride: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = shard_path(cache_dir, video, stride)
    if output.is_file():
        return output
    capture = cv2.VideoCapture(str(video.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video.video_path}")
    frame_indices, hashes, orientation_ids = [], [], []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            for orientation_id, transform in enumerate(ORIENTATIONS.values()):
                frame_indices.append(frame_index)
                hashes.append(dhash(transform(frame)))
                orientation_ids.append(orientation_id)
        frame_index += 1
    capture.release()
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        frame_index=np.asarray(frame_indices, dtype=np.int32),
        hash=np.asarray(hashes, dtype=np.uint64),
        orientation=np.asarray(orientation_ids, dtype=np.uint8),
    )
    temporary.replace(output)
    return output


def aggregate_index(videos: list[ReferenceVideo], cache_dir: Path, stride: int) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {"video_id": [], "frame_index": [], "hash": [], "orientation": []}
    for count, video in enumerate(videos, start=1):
        path = index_video(video, cache_dir, stride)
        with np.load(path) as shard:
            size = len(shard["hash"])
            arrays["video_id"].append(np.full(size, video.video_id, dtype=np.int32))
            for key in ("frame_index", "hash", "orientation"):
                arrays[key].append(shard[key])
        if count % 25 == 0 or count == len(videos):
            print(f"indexed_videos={count}/{len(videos)}", flush=True)
    return {key: np.concatenate(values) if values else np.empty(0) for key, values in arrays.items()}


def read_frame(path: Path, index: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(index, 0))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def read_frame_window(path: Path, start: int, end: int) -> dict[int, np.ndarray]:
    start = max(start, 0)
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = {}
    for index in range(start, end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        frames[index] = frame
    capture.release()
    return frames


def sequence_metrics(
    query_frames: list[np.ndarray],
    source_cache: dict[int, np.ndarray],
    end_index: int,
    orientation: str,
    source_stride: int,
) -> tuple[float, float] | None:
    source_frames = []
    for position in range(len(query_frames)):
        source_index = end_index - (len(query_frames) - 1 - position) * source_stride
        frame = source_cache.get(source_index)
        if frame is None:
            return None
        source_frames.append(frame)
    query_gray = [canonical(frame, "none") for frame in query_frames]
    source_gray = [canonical(frame, orientation) for frame in source_frames]
    mad = float(np.mean([pixel_mad(left, right) for left, right in zip(query_gray, source_gray)]))
    norm = float(np.mean([normalized_mad(left, right) for left, right in zip(query_gray, source_gray)]))
    return mad, norm


def shortlist(
    index: dict[str, np.ndarray],
    query_hash: np.uint64,
    references: list[ReferenceVideo],
    query_view: str,
    query_part: str,
    top_videos: int,
) -> list[tuple[int, int, int, int]]:
    compatible = np.array(
        [
            video.view_type == query_view
            or (video.dataset_part == query_part and query_part == "bdd_pc_5k")
            for video in references
        ],
        dtype=bool,
    )
    mask = compatible[index["video_id"]]
    positions = np.flatnonzero(mask)
    if not len(positions):
        return []
    distances = hamming_many(index["hash"][positions], query_hash)
    take = min(len(positions), max(top_videos * 20, top_videos))
    selected = positions[np.argpartition(distances, take - 1)[:take]]
    selected = selected[np.argsort(hamming_many(index["hash"][selected], query_hash))]
    per_video = []
    seen = set()
    for position in selected:
        video_id = int(index["video_id"][position])
        if video_id in seen:
            continue
        seen.add(video_id)
        per_video.append(
            (
                video_id,
                int(index["frame_index"][position]),
                int(index["orientation"][position]),
                int(hamming_many(index["hash"][position : position + 1], query_hash)[0]),
            )
        )
        if len(per_video) >= top_videos:
            break
    return per_video


def refine_query(
    row: dict[str, str],
    references: list[ReferenceVideo],
    index: dict[str, np.ndarray],
    coarse_stride: int,
    top_videos: int,
    sequence_length: int,
    sequence_strides: list[int],
) -> list[dict[str, object]]:
    paths = numeric_images(Path(row["input_dir"]))
    if not paths:
        return []
    query_frames = [cv2.imread(str(path)) for path in paths[-sequence_length:]]
    if any(frame is None for frame in query_frames):
        return []
    query_last = canonical(query_frames[-1], "none")
    candidates = shortlist(
        index,
        dhash(query_frames[-1]),
        references,
        row["view_type"],
        row["dataset_part"],
        top_videos,
    )
    results = []
    orientation_names = list(ORIENTATIONS)
    for video_id, coarse_frame, orientation_id, hash_distance in candidates:
        video = references[video_id]
        orientation = orientation_names[orientation_id]
        window_start = coarse_frame - coarse_stride - (len(query_frames) - 1) * max(sequence_strides)
        window_end = coarse_frame + coarse_stride
        source_cache = read_frame_window(video.video_path, window_start, window_end)
        best: dict[str, object] | None = None
        for end_index in range(max(0, coarse_frame - coarse_stride), coarse_frame + coarse_stride + 1):
            candidate_frame = source_cache.get(end_index)
            if candidate_frame is None:
                continue
            candidate_gray = canonical(candidate_frame, orientation)
            last_mad = pixel_mad(query_last, candidate_gray)
            last_norm = normalized_mad(query_last, candidate_gray)
            for source_stride in sequence_strides:
                metrics = sequence_metrics(query_frames, source_cache, end_index, orientation, source_stride)
                if metrics is None:
                    continue
                sequence_mad, sequence_norm = metrics
                rank_score = sequence_mad + 20.0 * sequence_norm + 0.25 * last_mad
                if best is None or rank_score < float(best["rank_score"]):
                    best = {
                        "query_sample_id": row["sample_id"],
                        "query_dataset_part": row["dataset_part"],
                        "query_view_type": row["view_type"],
                        "query_source_folder": row["source_folder"],
                        "reference_split": video.split,
                        "reference_dataset_part": video.dataset_part,
                        "reference_view_type": video.view_type,
                        "reference_source_folder": video.source_folder,
                        "reference_video": str(video.video_path),
                        "reference_end_frame": end_index,
                        "orientation": orientation,
                        "source_stride": source_stride,
                        "hash_distance": hash_distance,
                        "last_mad": last_mad,
                        "last_normalized_mad": last_norm,
                        "sequence_mad": sequence_mad,
                        "sequence_normalized_mad": sequence_norm,
                        "rank_score": rank_score,
                    }
        if best is not None:
            candidate_frame = source_cache[int(best["reference_end_frame"])]
            candidate_gray = canonical(candidate_frame, str(best["orientation"]))
            orb_matches, orb_ratio = orb_inlier_ratio(query_last, candidate_gray)
            best["orb_matches"] = orb_matches
            best["orb_inlier_ratio"] = orb_ratio
            best["exact_source_folder_match"] = row["source_folder"] == video.source_folder
            results.append(best)
    return sorted(results, key=lambda item: float(item["rank_score"]))


def make_sheet(row: dict[str, str], matches: list[dict[str, object]], output: Path, top_n: int) -> None:
    paths = numeric_images(Path(row["input_dir"]))
    query = cv2.imread(str(paths[-1]))
    if query is None:
        return
    query = cv2.resize(query, (384, 216), interpolation=cv2.INTER_AREA)
    tiles = []
    for rank, match in enumerate(matches[:top_n], start=1):
        candidate = read_frame(Path(str(match["reference_video"])), int(match["reference_end_frame"]))
        if candidate is None:
            continue
        candidate = ORIENTATIONS[str(match["orientation"])](candidate)
        candidate = cv2.resize(candidate, (384, 216), interpolation=cv2.INTER_AREA)
        left = query.copy()
        right = candidate.copy()
        cv2.putText(left, "query last", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(left, "query last", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        label = f"#{rank} mad={float(match['sequence_mad']):.1f} orb={float(match['orb_inlier_ratio']):.2f}"
        cv2.putText(right, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
        cv2.putText(right, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
        tiles.append(np.hstack([left, right]))
    if tiles:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), np.vstack(tiles))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_dir", type=Path, default=Path("/workspace/experiments/index"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--reference_splits", nargs="+", default=["train", "val"])
    parser.add_argument("--reference_stride", type=int, default=12)
    parser.add_argument("--top_videos", type=int, default=8)
    parser.add_argument("--sequence_length", type=int, default=5)
    parser.add_argument("--sequence_strides", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sheet_top_n", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    references = load_reference_videos(args.index_dir, args.reference_splits)
    (args.out_dir / "reference_videos.json").write_text(
        json.dumps([{**video.__dict__, "video_path": str(video.video_path)} for video in references], indent=2),
        encoding="utf-8",
    )
    print(f"reference_videos={len(references)}", flush=True)
    index = aggregate_index(references, args.out_dir / "index_cache", args.reference_stride)
    print(f"reference_hashes={len(index['hash'])}", flush=True)

    queries = [row for row in read_rows(args.index_dir, "test") if row.get("input_dir") and Path(row["input_dir"]).is_dir()]
    if args.limit is not None:
        queries = queries[: args.limit]
    all_matches = []
    summary = []
    for query_index, row in enumerate(queries, start=1):
        matches = refine_query(
            row,
            references,
            index,
            args.reference_stride,
            args.top_videos,
            args.sequence_length,
            args.sequence_strides,
        )
        all_matches.extend(matches)
        best = matches[0] if matches else None
        summary.append(
            {
                "query_sample_id": row["sample_id"],
                "query_dataset_part": row["dataset_part"],
                "query_view_type": row["view_type"],
                "best_reference_video": best["reference_video"] if best else "",
                "best_reference_end_frame": best["reference_end_frame"] if best else "",
                "best_sequence_mad": best["sequence_mad"] if best else "",
                "best_orb_inlier_ratio": best["orb_inlier_ratio"] if best else "",
                "exact_source_folder_match": best["exact_source_folder_match"] if best else "",
            }
        )
        make_sheet(row, matches, args.out_dir / "sheets" / f"{row['sample_id']}.jpg", args.sheet_top_n)
        print(f"queries={query_index}/{len(queries)} sample={row['sample_id']}", flush=True)

    write_csv(args.out_dir / "matches.csv", all_matches)
    write_csv(args.out_dir / "summary.csv", summary)
    print(f"summary={args.out_dir / 'summary.csv'}")
    print(f"matches={args.out_dir / 'matches.csv'}")
    print("Audit only: review metrics and sheets before considering any continuation extraction.")


if __name__ == "__main__":
    main()
