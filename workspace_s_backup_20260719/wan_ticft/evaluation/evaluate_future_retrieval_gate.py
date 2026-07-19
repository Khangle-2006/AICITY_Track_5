"""Evaluate train-set future retrieval as a high-ceiling video prediction route.

The gate answers one question: do train clips contain near-neighbor futures for
held-out validation clips?  If yes, retrieval/compositing can improve realism
much more than small postprocess tweaks.  If no, close this branch quickly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from baseline.video_io import read_frame_at


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2, paths: list[str]):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def read_indexed_frames(cv2, row: dict, indices: list[int]):
    video_path = Path(row.get("video_path", ""))
    if not video_path.exists() or not indices:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    for index in indices:
        bgr, cap = read_frame_at(cv2, video_path, cap, int(index))
        if bgr is not None:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_condition_frames(cv2, row: dict):
    paths = (row.get("frames") or {}).get("condition") or []
    frames = load_images(cv2, paths)
    if frames:
        return frames
    indices = (row.get("frame_indices") or {}).get("condition") or []
    return read_indexed_frames(cv2, row, [int(index) for index in indices])


def load_target_frames(cv2, row: dict):
    paths = (row.get("frames") or {}).get("target") or []
    frames = load_images(cv2, paths)
    if frames:
        return frames
    indices = (row.get("frame_indices") or {}).get("target") or []
    return read_indexed_frames(cv2, row, [int(index) for index in indices])


def condition_feature(cv2, np, row: dict, width: int, height: int):
    frames = load_condition_frames(cv2, row)
    if len(frames) < 2:
        return None
    small = []
    for image in frames[-4:]:
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        small.append(resized.astype(np.float32) / 255.0)
    while len(small) < 4:
        small.insert(0, small[0])
    diffs = [small[index] - small[index - 1] for index in range(1, len(small))]
    vector = np.concatenate([item.reshape(-1) for item in [*small, *diffs]])
    vector = vector.astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        vector /= norm
    return vector


def target_offsets(row: dict) -> list[int]:
    indices = row.get("frame_indices") or {}
    conditions = indices.get("condition") or []
    targets = indices.get("target") or []
    if not conditions or not targets:
        return list(range(1, len(targets) + 1))
    last = int(conditions[-1])
    return [max(0, int(index) - last) for index in targets]


def pick_train_targets(cv2, train_row: dict, eval_row: dict):
    train_frame_paths = (train_row.get("frames") or {}).get("target") or []
    train_indices = (train_row.get("frame_indices") or {}).get("target") or list(range(len(train_frame_paths)))
    train_conditions = (train_row.get("frame_indices") or {}).get("condition") or [0]
    if not train_frame_paths and not train_indices:
        return []
    train_last = int(train_conditions[-1])
    wanted = [train_last + offset for offset in target_offsets(eval_row)]
    picked = []
    for index in wanted:
        best_pos = min(range(len(train_indices)), key=lambda pos: abs(int(train_indices[pos]) - int(index)))
        image = read_rgb(cv2, Path(train_frame_paths[best_pos])) if train_frame_paths else None
        if image is None:
            frames = read_indexed_frames(cv2, train_row, [int(train_indices[best_pos])])
            image = frames[0] if frames else None
        if image is not None:
            picked.append(image)
    return picked


def linear_color_match(np, frames, train_row: dict, eval_row: dict, cv2):
    train_conditions = load_condition_frames(cv2, train_row)
    eval_conditions = load_condition_frames(cv2, eval_row)
    if not frames or not train_conditions or not eval_conditions:
        return frames
    source = train_conditions[-1].astype(np.float32)
    target = eval_conditions[-1].astype(np.float32)
    s_mean = source.reshape(-1, 3).mean(axis=0)
    s_std = source.reshape(-1, 3).std(axis=0) + 1e-3
    t_mean = target.reshape(-1, 3).mean(axis=0)
    t_std = target.reshape(-1, 3).std(axis=0) + 1e-3
    out = []
    for frame in frames:
        value = frame.astype(np.float32)
        value = (value - s_mean) * (t_std / s_std) + t_mean
        out.append(np.clip(value, 0, 255).astype(np.uint8))
    return out


def resize_like(cv2, frames, targets):
    out = []
    for frame, target in zip(frames, targets):
        if frame.shape[:2] != target.shape[:2]:
            frame = cv2.resize(frame, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)
        out.append(frame)
    return out


def score_sequence(cv2, np, frames, targets):
    values = [(psnr(cv2, np, p, t), ssim_gray(cv2, np, p, t), mean_absdiff(np, p, t)) for p, t in zip(frames, targets)]
    return tuple(float(np.mean([value[index] for value in values])) for index in range(3))


def write_video(cv2, frames, path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--view_type", action="append", default=None)
    parser.add_argument("--same_phase", action="store_true")
    parser.add_argument("--feature_width", type=int, default=48)
    parser.add_argument("--feature_height", type=int, default=27)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    view_types = set(args.view_type or [])
    train_rows = [row for row in read_jsonl(args.train_manifest) if not view_types or row.get("view_type") in view_types]
    eval_rows = [row for row in read_jsonl(args.eval_manifest) if not view_types or row.get("view_type") in view_types]
    if args.limit:
        eval_rows = eval_rows[: args.limit]

    index = []
    for position, row in enumerate(train_rows, start=1):
        feature = condition_feature(cv2, np, row, args.feature_width, args.feature_height)
        if feature is not None:
            index.append((row, feature))
        print(f"index [{position}/{len(train_rows)}] {row.get('sample_id')} | {'ok' if feature is not None else 'skip'}", flush=True)
    if not index:
        raise RuntimeError("empty retrieval index")

    report_rows = []
    detail_rows = []
    for position, row in enumerate(eval_rows, start=1):
        feature = condition_feature(cv2, np, row, args.feature_width, args.feature_height)
        targets = load_target_frames(cv2, row)
        conditions = load_condition_frames(cv2, row)
        if feature is None or not targets or not conditions:
            continue
        pool = [(train_row, train_feature) for train_row, train_feature in index if train_row.get("view_type") == row.get("view_type")]
        if args.same_phase:
            phased = [(train_row, train_feature) for train_row, train_feature in pool if str(train_row.get("phase", "")) == str(row.get("phase", ""))]
            if phased:
                pool = phased
        distances = sorted(
            ((float(np.mean((feature - train_feature) ** 2)), train_row) for train_row, train_feature in pool),
            key=lambda item: item[0],
        )
        for rank, (distance, train_row) in enumerate(distances[: args.topk], start=1):
            retrieved = pick_train_targets(cv2, train_row, row)
            retrieved = resize_like(cv2, retrieved, targets)
            if len(retrieved) != len(targets):
                continue
            matched = resize_like(cv2, linear_color_match(np, retrieved, train_row, row, cv2), targets)
            repeat = [conditions[-1]] * len(targets)
            repeat = resize_like(cv2, repeat, targets)
            for candidate, frames in (("repeat_last", repeat), (f"retrieval_top{rank}", retrieved), (f"retrieval_top{rank}_cm", matched)):
                p, s, d = score_sequence(cv2, np, frames, targets)
                detail_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "view_type": row.get("view_type", ""),
                        "phase": row.get("phase", ""),
                        "candidate": candidate,
                        "rank": rank,
                        "distance": f"{distance:.8f}",
                        "train_sample_id": train_row["sample_id"],
                        "psnr": f"{p:.6f}",
                        "ssim": f"{s:.6f}",
                        "absdiff": f"{d:.6f}",
                    }
                )
            if rank == 1:
                video_dir = args.out_dir / "videos" / "retrieval_top1"
                write_video(cv2, retrieved, video_dir / f"{row['sample_id']}.avi", 30.0)
                write_video(cv2, matched, args.out_dir / "videos" / "retrieval_top1_cm" / f"{row['sample_id']}.avi", 30.0)
                report_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "view_type": row.get("view_type", ""),
                        "phase": row.get("phase", ""),
                        "video": str(video_dir / f"{row['sample_id']}.avi"),
                        "train_sample_id": train_row["sample_id"],
                        "distance": f"{distance:.8f}",
                        "future_frames": len(retrieved),
                    }
                )
        print(f"eval [{position}/{len(eval_rows)}] {row.get('sample_id')} | neighbors={min(len(distances), args.topk)}", flush=True)

    with (args.out_dir / "eval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_rows[0].keys())
        writer.writeheader()
        writer.writerows(detail_rows)
    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    print()
    print(f"{'Candidate':<22} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 62)
    for candidate in sorted({row["candidate"] for row in detail_rows}):
        items = [row for row in detail_rows if row["candidate"] == candidate]
        print(
            f"{candidate:<22} {len(items):>5} "
            f"{np.mean([float(x['psnr']) for x in items]):>10.4f} "
            f"{np.mean([float(x['ssim']) for x in items]):>10.4f} "
            f"{np.mean([float(x['absdiff']) for x in items]):>10.4f}"
        )
    print(f"output {args.out_dir}")


if __name__ == "__main__":
    main()
