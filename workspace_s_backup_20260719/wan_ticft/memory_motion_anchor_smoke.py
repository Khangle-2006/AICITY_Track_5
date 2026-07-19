"""Memory-retrieved motion anchors for a BDD test sample.

This is a lightweight adaptation of Memory-V2V's idea: retrieve relevant past
videos, use them as structured motion constraints, but do not paste their pixels.
For each future offset, we estimate an affine motion from the retrieved sample's
last condition frame to its target frame, then apply that transform to the test
sample's own last observed frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


def numeric_images(directory: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        match = re.search(r"(\d+)(?!.*\d)", path.stem)
        return (int(match.group(1)) if match else -1, path.name)

    return sorted([p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}], key=key)


def read_rgb(path: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_video_indices(video: Path, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_memory_from_index(index_paths: list[Path], limit: int) -> list[tuple[dict, list[np.ndarray], list[np.ndarray]]]:
    rows: list[tuple[dict, list[np.ndarray], list[np.ndarray]]] = []
    for index_path in index_paths:
        for row in read_csv(index_path):
            if row.get("view_type") != "bdd" or not row.get("video_path"):
                continue
            video = Path(row["video_path"])
            if not video.exists():
                continue
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                continue
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if total <= 20:
                continue
            try:
                end_time = float(row.get("end_time") or row.get("start_time") or 0.0)
            except ValueError:
                continue
            end_frame = int(round(end_time * fps))
            end_frame = min(max(end_frame, 3), total - 10)
            start_frame = max(0, end_frame - 40)
            cond_indices = np.linspace(start_frame, end_frame, 4).round().astype(int).tolist()
            target_indices = [min(total - 1, end_frame + 1 + i) for i in range(9)]
            cond = read_video_indices(video, cond_indices)
            targets = read_video_indices(video, target_indices)
            if len(cond) == 4 and len(targets) == 9:
                rows.append((row, cond, targets))
                if limit and len(rows) >= limit:
                    return rows
    return rows


def load_row_future(row: dict[str, str], target_n: int, fallback: list[np.ndarray]) -> list[np.ndarray]:
    video_raw = row.get("video_path")
    if not video_raw:
        return fallback
    video = Path(video_raw)
    if not video.exists():
        return fallback
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return fallback
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    try:
        end_time = float(row.get("end_time") or row.get("start_time") or 0.0)
    except ValueError:
        return fallback
    end_frame = int(round(end_time * fps))
    end_frame = min(max(end_frame, 3), max(0, total - 2))
    target_indices = [min(total - 1, end_frame + 1 + i) for i in range(target_n)]
    targets = read_video_indices(video, target_indices)
    return targets if len(targets) >= min(target_n, len(fallback)) else fallback


def load_frames(paths: list[str]) -> list[np.ndarray]:
    frames = []
    for raw in paths:
        image = read_rgb(Path(raw))
        if image is not None:
            frames.append(image)
    return frames


def load_test_input(index_csv: Path, sample_substring: str) -> tuple[dict[str, str], list[np.ndarray]]:
    rows = [row for row in read_csv(index_csv) if row.get("view_type") == "bdd" and sample_substring in row.get("source_folder", "")]
    if not rows:
        raise RuntimeError(f"no BDD test row matching {sample_substring}")
    row = rows[0]
    paths = numeric_images(Path(row["input_dir"]))
    frames = [read_rgb(path) for path in paths]
    frames = [frame for frame in frames if frame is not None]
    if len(frames) < 4:
        raise RuntimeError(f"{sample_substring}: need >=4 input frames, got {len(frames)}")
    return row, frames


def feature(frames: list[np.ndarray], width: int = 64, height: int = 36) -> np.ndarray:
    selected = frames[-4:]
    small = []
    for image in selected:
        small.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
    diffs = [small[i] - small[i - 1] for i in range(1, len(small))]
    gray = [cv2.cvtColor((s * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for s in small]
    edges = [cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) for g in gray[-2:]]
    vec = np.concatenate([x.reshape(-1) for x in [*small, *diffs, *edges]]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-8)


def estimate_affine(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, int]:
    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=7)
    kp1, des1 = orb.detectAndCompute(src_gray, None)
    kp2, des2 = orb.detectAndCompute(dst_gray, None)
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return identity, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [a for a, b in matches if a.distance < 0.75 * b.distance]
    if len(good) < 8:
        return identity, len(good)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    mat, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=3000)
    if mat is None:
        return identity, len(good)
    count = int(inliers.sum()) if inliers is not None else len(good)
    # Clamp extreme retrieval accidents. Dashcam future motion should be modest.
    tx, ty = float(mat[0, 2]), float(mat[1, 2])
    if abs(tx) > src.shape[1] * 0.15 or abs(ty) > src.shape[0] * 0.15:
        return identity, count
    return mat.astype(np.float32), count


def warp_frame(image: np.ndarray, mat: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def ghost_weight(base: np.ndarray, anchor: np.ndarray, strength: float) -> np.ndarray:
    diff = np.mean(np.abs(base.astype(np.float32) - anchor.astype(np.float32)), axis=2)
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dark = np.clip((105.0 - gray) / 105.0, 0.0, 1.0)
    mask = np.clip((diff - 10.0) / 45.0, 0.0, 1.0) * dark
    mask = cv2.GaussianBlur(mask, (0, 0), 5.0)
    return np.clip(mask * strength, 0.0, 0.45).astype(np.float32)


def clamp_weight(base: np.ndarray, anchor: np.ndarray, strength: float) -> np.ndarray:
    """Strongly replace geometry-uncertain dark trails with the memory anchor."""
    diff = np.mean(np.abs(base.astype(np.float32) - anchor.astype(np.float32)), axis=2)
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dark = np.clip((130.0 - gray) / 130.0, 0.0, 1.0)
    uncertain = np.maximum(np.clip((diff - 18.0) / 38.0, 0.0, 1.0) * dark, np.clip((diff - 42.0) / 70.0, 0.0, 1.0))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    uncertain = cv2.dilate(uncertain.astype(np.float32), kernel, iterations=1)
    uncertain = cv2.GaussianBlur(uncertain, (0, 0), 3.0)
    return np.clip(uncertain * strength, 0.0, 0.92).astype(np.float32)


def memory_group_key(row: dict) -> str:
    sample_id = str(row.get("sample_id") or row.get("source_folder") or "")
    return re.sub(r"_phase\d+$", "", sample_id)


def select_diverse(scored: list[tuple[float, dict, list[np.ndarray], list[np.ndarray]]], topk: int) -> list[tuple[float, dict, list[np.ndarray], list[np.ndarray]]]:
    selected = []
    seen: set[str] = set()
    for item in scored:
        key = memory_group_key(item[1])
        if key in seen:
            continue
        selected.append(item)
        seen.add(key)
        if len(selected) >= topk:
            return selected
    return selected or scored[:topk]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_index", type=Path, default=Path("/workspace/experiments/index/test_index.csv"))
    parser.add_argument("--memory_manifest", type=Path, default=Path("/workspace/experiments/wan_ticft/manifests/train_ticft_balanced_300_frames.jsonl"))
    parser.add_argument("--memory_index_csv", type=Path, action="append", default=[])
    parser.add_argument("--memory_index_limit", type=int, default=600)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--sample", default="video3806")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--blend_strength", type=float, default=0.35)
    parser.add_argument("--anchor_offsets", type=int, default=9)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2.setNumThreads(4)
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    test_row, test_history = load_test_input(args.test_index, args.sample)
    query = feature(test_history)
    memory_rows = []
    if args.memory_index_csv:
        source_rows = load_memory_from_index(args.memory_index_csv, args.memory_index_limit)
        memory_rows = [(row, cond, targets, feature(cond)) for row, cond, targets in source_rows]
    else:
        for row in read_jsonl(args.memory_manifest):
            if row.get("view_type") != "bdd":
                continue
            cond = load_frames((row.get("frames") or {}).get("condition") or [])
            targets = load_frames((row.get("frames") or {}).get("target") or [])
            if len(cond) >= 4 and len(targets) >= 9:
                memory_rows.append((row, cond, targets, feature(cond)))
    scored = sorted(((float(np.mean((query - feat) ** 2)), row, cond, targets) for row, cond, targets, feat in memory_rows), key=lambda x: x[0])
    neighbors = select_diverse(scored, args.topk)
    if not neighbors:
        raise RuntimeError("empty memory neighbors")

    target_n = int(test_row["target_N"])
    base_paths = numeric_images(args.base_dir / args.sample)
    base_frames = [read_rgb(path) for path in base_paths[:target_n]]
    base_frames = [frame for frame in base_frames if frame is not None]
    if len(base_frames) != target_n:
        raise RuntimeError(f"base frames {len(base_frames)} != target {target_n}")

    last = test_history[-1]
    if last.shape[:2] != base_frames[0].shape[:2]:
        last = cv2.resize(last, (base_frames[0].shape[1], base_frames[0].shape[0]), interpolation=cv2.INTER_AREA)

    rows = []
    mats_by_offset: dict[int, list[tuple[float, np.ndarray, int, str]]] = {}
    for rank, (dist, row, cond, targets) in enumerate(neighbors, start=1):
        src = cond[-1]
        targets = load_row_future(row, min(target_n, args.anchor_offsets), targets)
        for offset in range(min(target_n, args.anchor_offsets, len(targets))):
            dst = targets[offset]
            if dst.shape[:2] != src.shape[:2]:
                dst = cv2.resize(dst, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
            mat, inliers = estimate_affine(src, dst)
            mats_by_offset.setdefault(offset, []).append((dist, mat, inliers, row["sample_id"]))
        rows.append({"rank": rank, "distance": f"{dist:.8f}", "sample_id": row["sample_id"]})

    anchor_dir = args.out_dir / "memory_anchor"
    blend_dir = args.out_dir / "memory_blend"
    clamp_dir = args.out_dir / "memory_clamp"
    for index, base in enumerate(base_frames):
        if index < args.anchor_offsets and index in mats_by_offset:
            # Weighted average affine parameters from top retrieved memories.
            items = mats_by_offset[index]
            weights = np.array([1.0 / (dist + 1e-6) for dist, _, _, _ in items], dtype=np.float32)
            weights = weights / weights.sum()
            mat = sum(float(w) * m for w, (_, m, _, _) in zip(weights, items)).astype(np.float32)
            anchor = warp_frame(last, mat)
            if anchor.shape[:2] != base.shape[:2]:
                anchor = cv2.resize(anchor, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
            weight = ghost_weight(base, anchor, args.blend_strength)[..., None]
            blend = np.clip(base.astype(np.float32) * (1.0 - weight) + anchor.astype(np.float32) * weight, 0, 255).astype(np.uint8)
            clamp = np.clip(base.astype(np.float32) * (1.0 - clamp_weight(base, anchor, args.blend_strength)[..., None]) + anchor.astype(np.float32) * clamp_weight(base, anchor, args.blend_strength)[..., None], 0, 255).astype(np.uint8)
        else:
            anchor = base
            blend = base
            clamp = base
        write_rgb(anchor_dir / f"{index}.png", anchor)
        write_rgb(blend_dir / f"{index}.png", blend)
        write_rgb(clamp_dir / f"{index}.png", clamp)

    with (args.out_dir / "neighbors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "distance", "sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "motion.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["offset", "rank", "distance", "inliers", "sample_id", "a00", "a01", "a02", "a10", "a11", "a12"])
        writer.writeheader()
        for offset, items in sorted(mats_by_offset.items()):
            for rank, (dist, mat, inliers, sid) in enumerate(items, start=1):
                writer.writerow({
                    "offset": offset,
                    "rank": rank,
                    "distance": f"{dist:.8f}",
                    "inliers": inliers,
                    "sample_id": sid,
                    "a00": f"{mat[0,0]:.6f}",
                    "a01": f"{mat[0,1]:.6f}",
                    "a02": f"{mat[0,2]:.3f}",
                    "a10": f"{mat[1,0]:.6f}",
                    "a11": f"{mat[1,1]:.6f}",
                    "a12": f"{mat[1,2]:.3f}",
                })
    print(f"output={args.out_dir}")
    print(f"neighbors={len(rows)} target_N={target_n}")


if __name__ == "__main__":
    main()
