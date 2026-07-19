"""Measure a global-transform plus local-residual-flow oracle on BDD validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.generators import compute_flow, warp_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.evaluation.evaluate_bdd_ego_motion import estimate_transform


CANDIDATES = (
    "repeat_last",
    "oracle_global_choice",
    "oracle_direct_flow",
    "oracle_layered_flow",
    "oracle_layered_block64",
    "oracle_layered_block32",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2, paths):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def error(np, prediction, target):
    return float(np.mean(np.abs(prediction.astype(np.float32) - target.astype(np.float32))))


def flow_magnitude(np, flow):
    return float(np.mean(np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)))


def block_oracle(np, first, second, target, block: int):
    output = first.copy()
    height, width = target.shape[:2]
    for y in range(0, height, block):
        for x in range(0, width, block):
            ys = slice(y, min(height, y + block))
            xs = slice(x, min(width, x + block))
            target_patch = target[ys, xs].astype(np.float32)
            first_error = np.mean(np.abs(first[ys, xs].astype(np.float32) - target_patch))
            second_error = np.mean(np.abs(second[ys, xs].astype(np.float32) - target_patch))
            if second_error < first_error:
                output[ys, xs] = second[ys, xs]
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
    diagnostics = []
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
        direct_magnitudes = []
        residual_magnitudes = []
        for target in targets:
            global_predictions = []
            for model in ("partial", "affine", "homography"):
                transform, _, _ = estimate_transform(
                    cv2, np, last, target, model, args.estimate_width, args.roi_bottom
                )
                global_predictions.append(
                    cv2.warpPerspective(
                        last,
                        transform,
                        (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                )
            global_choice = min(global_predictions, key=lambda prediction: error(np, prediction, target))

            direct_flow = compute_flow(cv2, np, target, last, args.estimate_width)
            direct_prediction = warp_rgb(cv2, np, last, direct_flow, -1.0)

            residual_flow = compute_flow(cv2, np, target, global_choice, args.estimate_width)
            layered_prediction = warp_rgb(cv2, np, global_choice, residual_flow, -1.0)

            direct_magnitudes.append(flow_magnitude(np, direct_flow))
            residual_magnitudes.append(flow_magnitude(np, residual_flow))
            candidates["repeat_last"].append(last.copy())
            candidates["oracle_global_choice"].append(global_choice)
            candidates["oracle_direct_flow"].append(direct_prediction)
            candidates["oracle_layered_flow"].append(layered_prediction)
            candidates["oracle_layered_block64"].append(block_oracle(np, global_choice, layered_prediction, target, 64))
            candidates["oracle_layered_block32"].append(block_oracle(np, global_choice, layered_prediction, target, 32))

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
        diagnostics.append(
            {
                "sample_id": row["sample_id"],
                "phase": row.get("phase", ""),
                "direct_flow_magnitude": f"{np.mean(direct_magnitudes):.6f}",
                "residual_flow_magnitude": f"{np.mean(residual_magnitudes):.6f}",
                "residual_ratio": f"{np.mean(residual_magnitudes) / max(np.mean(direct_magnitudes), 1e-6):.6f}",
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
    diagnostics_path = args.out_csv.with_name(args.out_csv.stem + "_diagnostics.csv")
    with diagnostics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostics[0].keys())
        writer.writeheader()
        writer.writerows(diagnostics)

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
    print()
    print(f"mean direct flow magnitude       {np.mean([float(row['direct_flow_magnitude']) for row in diagnostics]):.4f}")
    print(f"mean residual flow magnitude     {np.mean([float(row['residual_flow_magnitude']) for row in diagnostics]):.4f}")
    print(f"mean residual/direct ratio       {np.mean([float(row['residual_ratio']) for row in diagnostics]):.4f}")
    print(f"output {args.out_csv}")
    print(f"diagnostics {diagnostics_path}")


if __name__ == "__main__":
    main()
