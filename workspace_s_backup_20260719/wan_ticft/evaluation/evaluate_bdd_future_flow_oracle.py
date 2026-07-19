"""Measure the rendering ceiling of perfect future-flow knowledge on BDD validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.generators import compute_flow, warp_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray


CANDIDATES = (
    "repeat_last",
    "oracle_forward_flow",
    "oracle_backward_flow",
    "oracle_direction",
    "oracle_block64",
    "oracle_block32",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2, paths: list[str]):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def score(cv2, np, predictions, targets):
    values = [
        (
            psnr(cv2, np, prediction, target),
            ssim_gray(cv2, np, prediction, target),
            mean_absdiff(np, prediction, target),
        )
        for prediction, target in zip(predictions, targets)
    ]
    return tuple(float(np.mean([value[index] for value in values])) for index in range(3))


def abs_error(np, prediction, target):
    return float(np.mean(np.abs(prediction.astype(np.float32) - target.astype(np.float32))))


def block_oracle(np, repeat, flow_prediction, target, block: int):
    output = repeat.copy()
    height, width = target.shape[:2]
    for y in range(0, height, block):
        for x in range(0, width, block):
            ys = slice(y, min(height, y + block))
            xs = slice(x, min(width, x + block))
            target_patch = target[ys, xs].astype(np.float32)
            repeat_error = np.mean(np.abs(repeat[ys, xs].astype(np.float32) - target_patch))
            flow_error = np.mean(np.abs(flow_prediction[ys, xs].astype(np.float32) - target_patch))
            if flow_error < repeat_error:
                output[ys, xs] = flow_prediction[ys, xs]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--flow_width", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    results = []
    processed = 0
    for position, row in enumerate(read_jsonl(args.manifest), start=1):
        if row.get("view_type") != "bdd":
            continue
        if args.limit and processed >= args.limit:
            break
        paths = row.get("frames") or {}
        conditions = load_images(cv2, paths.get("condition") or [])
        targets = load_images(cv2, paths.get("target") or [])
        if not conditions or not targets:
            print(f"[{position}] {row['sample_id']} | skipped: missing frames", flush=True)
            continue

        last = conditions[-1]
        candidates = {name: [] for name in CANDIDATES}
        for target in targets:
            if last.shape[:2] != target.shape[:2]:
                last = cv2.resize(last, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_AREA)

            # Forward flow describes where pixels from the final observation move.
            forward = compute_flow(cv2, np, last, target, args.flow_width)
            forward_prediction = warp_rgb(cv2, np, last, forward, 1.0)

            # Backward flow directly estimates the source location for each target pixel.
            backward = compute_flow(cv2, np, target, last, args.flow_width)
            backward_prediction = warp_rgb(cv2, np, last, backward, -1.0)

            direction_prediction = min(
                (forward_prediction, backward_prediction),
                key=lambda prediction: abs_error(np, prediction, target),
            )
            candidates["repeat_last"].append(last.copy())
            candidates["oracle_forward_flow"].append(forward_prediction)
            candidates["oracle_backward_flow"].append(backward_prediction)
            candidates["oracle_direction"].append(direction_prediction)
            candidates["oracle_block64"].append(block_oracle(np, last, direction_prediction, target, 64))
            candidates["oracle_block32"].append(block_oracle(np, last, direction_prediction, target, 32))

        for candidate, predictions in candidates.items():
            p, s, d = score(cv2, np, predictions, targets)
            results.append(
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

    if not results:
        raise RuntimeError("no BDD samples were evaluated")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"{'Candidate':<26} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 66)
    for candidate in CANDIDATES:
        rows = [row for row in results if row["candidate"] == candidate]
        print(
            f"{candidate:<26} {len(rows):>5} "
            f"{np.mean([float(row['psnr']) for row in rows]):>10.4f} "
            f"{np.mean([float(row['ssim']) for row in rows]):>10.4f} "
            f"{np.mean([float(row['absdiff']) for row in rows]):>10.4f}"
        )
    print(f"output {args.out_csv}")


if __name__ == "__main__":
    main()
