"""Measure the ceiling of future global camera transforms on BDD validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.evaluation.evaluate_bdd_ego_motion import estimate_transform


CANDIDATES = (
    "repeat_last",
    "oracle_partial",
    "oracle_affine",
    "oracle_homography",
    "oracle_global_choice",
    "oracle_global_block64",
    "oracle_global_block32",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2, paths):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def abs_error(np, prediction, target):
    return float(np.mean(np.abs(prediction.astype(np.float32) - target.astype(np.float32))))


def block_oracle(np, repeat, transformed, target, block: int):
    output = repeat.copy()
    height, width = target.shape[:2]
    for y in range(0, height, block):
        for x in range(0, width, block):
            ys = slice(y, min(height, y + block))
            xs = slice(x, min(width, x + block))
            target_patch = target[ys, xs].astype(np.float32)
            repeat_error = np.mean(np.abs(repeat[ys, xs].astype(np.float32) - target_patch))
            transformed_error = np.mean(np.abs(transformed[ys, xs].astype(np.float32) - target_patch))
            if transformed_error < repeat_error:
                output[ys, xs] = transformed[ys, xs]
    return output


def score(cv2, np, predictions, targets):
    metrics = [
        (
            psnr(cv2, np, prediction, target),
            ssim_gray(cv2, np, prediction, target),
            mean_absdiff(np, prediction, target),
        )
        for prediction, target in zip(predictions, targets)
    ]
    return tuple(float(np.mean([value[index] for value in metrics])) for index in range(3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--estimate_width", type=int, default=640)
    parser.add_argument("--roi_bottom", type=float, default=0.85)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows_out = []
    processed = 0
    for row in read_jsonl(args.manifest):
        if row.get("view_type") != "bdd":
            continue
        paths = row.get("frames") or {}
        conditions = load_images(cv2, paths.get("condition") or [])
        targets = load_images(cv2, paths.get("target") or [])
        if not conditions or not targets:
            continue
        last = conditions[-1]
        height, width = last.shape[:2]
        candidates = {name: [] for name in CANDIDATES}
        match_stats = []
        for target in targets:
            transformed = {}
            stats = {}
            for model in ("partial", "affine", "homography"):
                transform, matches, inlier_ratio = estimate_transform(
                    cv2, np, last, target, model, args.estimate_width, args.roi_bottom
                )
                transformed[model] = cv2.warpPerspective(
                    last,
                    transform,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                stats[model] = (matches, inlier_ratio)
            choice = min(transformed.values(), key=lambda prediction: abs_error(np, prediction, target))
            candidates["repeat_last"].append(last.copy())
            candidates["oracle_partial"].append(transformed["partial"])
            candidates["oracle_affine"].append(transformed["affine"])
            candidates["oracle_homography"].append(transformed["homography"])
            candidates["oracle_global_choice"].append(choice)
            candidates["oracle_global_block64"].append(block_oracle(np, last, choice, target, 64))
            candidates["oracle_global_block32"].append(block_oracle(np, last, choice, target, 32))
            match_stats.append(stats)

        for candidate, predictions in candidates.items():
            p, s, d = score(cv2, np, predictions, targets)
            rows_out.append(
                {
                    "sample_id": row["sample_id"],
                    "phase": row.get("phase", ""),
                    "candidate": candidate,
                    "target_count": len(targets),
                    "psnr": f"{p:.6f}",
                    "ssim": f"{s:.6f}",
                    "absdiff": f"{d:.6f}",
                }
            )
        processed += 1
        print(f"[{processed}] {row['sample_id']} | ok", flush=True)

    if not rows_out:
        raise RuntimeError("no BDD samples evaluated")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)

    print()
    print(f"{'Candidate':<26} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 66)
    for candidate in CANDIDATES:
        items = [row for row in rows_out if row["candidate"] == candidate]
        print(
            f"{candidate:<26} {len(items):>5} "
            f"{np.mean([float(row['psnr']) for row in items]):>10.4f} "
            f"{np.mean([float(row['ssim']) for row in items]):>10.4f} "
            f"{np.mean([float(row['absdiff']) for row in items]):>10.4f}"
        )
    print(f"output {args.out_csv}")


if __name__ == "__main__":
    main()
