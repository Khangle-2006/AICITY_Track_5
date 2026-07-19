"""Cross-validate a learned spatial selector between LTX and region predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_bdd_ego_motion import region_predictions, score_sequence
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_images(cv2, paths: list[str]):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def target_offsets(row: dict, count: int) -> list[int]:
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    target = indices.get("target") or []
    if not condition or len(target) != count:
        return list(range(1, count + 1))
    last = int(condition[-1])
    return [max(1, int(index) - last) for index in target]


def resize_like(cv2, image, target):
    if image.shape[:2] == target.shape[:2]:
        return image
    return cv2.resize(image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)


def patch_features(cv2, np, region, ltx, last, previous, x, y, block, frame_index, frame_count):
    ys = slice(y, min(region.shape[0], y + block))
    xs = slice(x, min(region.shape[1], x + block))
    r = region[ys, xs].astype(np.float32)
    l = ltx[ys, xs].astype(np.float32)
    a = last[ys, xs].astype(np.float32)
    p = previous[ys, xs].astype(np.float32)
    gray_r = cv2.cvtColor(region[ys, xs], cv2.COLOR_RGB2GRAY)
    gray_l = cv2.cvtColor(ltx[ys, xs], cv2.COLOR_RGB2GRAY)
    return [
        x / region.shape[1],
        y / region.shape[0],
        frame_index / max(1, frame_count - 1),
        float(np.mean(np.abs(r - l))) / 255.0,
        float(np.mean(np.abs(r - a))) / 255.0,
        float(np.mean(np.abs(l - a))) / 255.0,
        float(np.mean(np.abs(a - p))) / 255.0,
        float(np.std(r)) / 255.0,
        float(np.std(l)) / 255.0,
        float(cv2.Laplacian(gray_r, cv2.CV_32F).var()) / 10000.0,
        float(cv2.Laplacian(gray_l, cv2.CV_32F).var()) / 10000.0,
    ]


def collect_sample(cv2, np, row, video, block, estimate_width):
    paths = row.get("frames") or {}
    conditions = load_images(cv2, paths.get("condition") or [])
    targets = load_images(cv2, paths.get("target") or [])
    if len(conditions) < 2 or not targets:
        return None
    indices = (row.get("frame_indices") or {}).get("condition") or []
    history_gap = max(1, int(indices[-1]) - int(indices[-2])) if len(indices) >= 2 else 1
    offsets = target_offsets(row, len(targets))
    region = region_predictions(cv2, np, conditions[-2], conditions[-1], offsets, history_gap, estimate_width)
    mapped = map_target_offsets(row, video_fps(cv2, video))
    frame_map = read_video_rgb_frames(cv2, video, mapped)
    ltx = [frame_map[index] for index in mapped if index in frame_map]
    if len(ltx) != len(targets):
        return None
    features, labels, patches = [], [], []
    normalized_region, normalized_ltx = [], []
    for frame_index, (r, l, target) in enumerate(zip(region, ltx, targets)):
        r = resize_like(cv2, r, target)
        l = resize_like(cv2, l, target)
        normalized_region.append(r)
        normalized_ltx.append(l)
        last = resize_like(cv2, conditions[-1], target)
        previous = resize_like(cv2, conditions[-2], target)
        for y in range(0, target.shape[0], block):
            for x in range(0, target.shape[1], block):
                ys = slice(y, min(target.shape[0], y + block))
                xs = slice(x, min(target.shape[1], x + block))
                target_patch = target[ys, xs].astype(np.float32)
                region_error = float(np.mean(np.abs(r[ys, xs].astype(np.float32) - target_patch)))
                ltx_error = float(np.mean(np.abs(l[ys, xs].astype(np.float32) - target_patch)))
                features.append(
                    patch_features(
                        cv2, np, r, l, last, previous, x, y, block, frame_index, len(targets)
                    )
                )
                labels.append(int(ltx_error < region_error))
                patches.append((frame_index, x, y))
    return {
        "sample_id": row["sample_id"],
        "features": np.asarray(features, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "patches": patches,
        "region": normalized_region,
        "ltx": normalized_ltx,
        "targets": targets,
    }


def composite(np, sample, predictions, block):
    frames = [frame.copy() for frame in sample["region"]]
    for use_ltx, (frame_index, x, y) in zip(predictions, sample["patches"]):
        if use_ltx:
            ys = slice(y, min(frames[frame_index].shape[0], y + block))
            xs = slice(x, min(frames[frame_index].shape[1], x + block))
            frames[frame_index][ys, xs] = sample["ltx"][frame_index][ys, xs]
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--estimate_width", type=int, default=640)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    from sklearn.ensemble import HistGradientBoostingClassifier

    reports = read_report(args.ltx_report)
    samples = []
    for position, row in enumerate(read_jsonl(args.manifest), start=1):
        video = reports.get(row["sample_id"])
        sample = collect_sample(cv2, np, row, video, args.block, args.estimate_width) if video else None
        if sample:
            samples.append(sample)
            print(f"[{position}] {row['sample_id']} | prepared", flush=True)

    results = []
    midpoint = len(samples) // 2
    for fold, (train, test) in enumerate(((samples[:midpoint], samples[midpoint:]), (samples[midpoint:], samples[:midpoint])), start=1):
        train_x = np.concatenate([sample["features"] for sample in train])
        train_y = np.concatenate([sample["labels"] for sample in train])
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=120,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=17,
        )
        model.fit(train_x, train_y)
        for sample in test:
            predicted = model.predict(sample["features"])
            learned = composite(np, sample, predicted, args.block)
            oracle = composite(np, sample, sample["labels"], args.block)
            accuracy = float(np.mean(predicted == sample["labels"]))
            for candidate, frames in (
                ("region", sample["region"]),
                ("ltx16", sample["ltx"]),
                ("learned_selector", learned),
                ("block_oracle", oracle),
            ):
                p, s, d = score_sequence(cv2, np, frames, sample["targets"])
                results.append(
                    {
                        "fold": fold,
                        "sample_id": sample["sample_id"],
                        "candidate": candidate,
                        "selector_accuracy": f"{accuracy:.6f}",
                        "ltx_patch_fraction": f"{float(np.mean(predicted)):.6f}",
                        "psnr": f"{p:.6f}",
                        "ssim": f"{s:.6f}",
                        "absdiff": f"{d:.6f}",
                    }
                )
            print(f"fold {fold} | {sample['sample_id']} | accuracy={accuracy:.3f}", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"{'Candidate':<20} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 60)
    for candidate in ("region", "ltx16", "learned_selector", "block_oracle"):
        rows = [row for row in results if row["candidate"] == candidate]
        print(
            f"{candidate:<20} {len(rows):>5} "
            f"{np.mean([float(x['psnr']) for x in rows]):>10.4f} "
            f"{np.mean([float(x['ssim']) for x in rows]):>10.4f} "
            f"{np.mean([float(x['absdiff']) for x in rows]):>10.4f}"
        )
    selector_rows = [row for row in results if row["candidate"] == "learned_selector"]
    print(f"selector accuracy: {np.mean([float(x['selector_accuracy']) for x in selector_rows]):.4f}")
    print(f"output {args.out_csv}")


if __name__ == "__main__":
    main()
