#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_epona_one_frame import ssim_gray


def metrics(image, reference):
    image = cv2.resize(
        image,
        (reference.shape[1], reference.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    diff = image.astype(np.float32) - reference.astype(np.float32)
    mse = float(np.mean(diff * diff))
    return {
        "psnr": float(10.0 * np.log10((255.0 * 255.0) / max(mse, 1e-12))),
        "ssim": ssim_gray(reference, image),
        "absdiff": float(np.mean(np.abs(diff))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--reference_root", required=True)
    parser.add_argument("--prediction_root", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    rows = []
    sample_dirs = sorted(path for path in Path(args.data_root).iterdir() if path.is_dir())
    for index, sample_dir in enumerate(sample_dirs):
        sample_id = sample_dir.name
        pred = cv2.imread(str(Path(args.prediction_root) / f"sliding_{index}" / "10.png"))
        baseline = cv2.imread(str(sample_dir / "000009.png"))
        reference = cv2.imread(str(Path(args.reference_root) / f"{sample_id}.png"))
        if pred is None or baseline is None or reference is None:
            raise RuntimeError(f"missing image for {sample_id}")
        epona = metrics(pred, reference)
        repeat = metrics(baseline, reference)
        rows.append(
            {
                "sample_id": sample_id,
                "epona_psnr": epona["psnr"],
                "repeat_psnr": repeat["psnr"],
                "delta_psnr": epona["psnr"] - repeat["psnr"],
                "epona_ssim": epona["ssim"],
                "repeat_ssim": repeat["ssim"],
                "delta_ssim": epona["ssim"] - repeat["ssim"],
                "epona_absdiff": epona["absdiff"],
                "repeat_absdiff": repeat["absdiff"],
            }
        )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "samples": len(rows),
        "wins": sum(row["delta_psnr"] > 0 for row in rows),
        "epona_psnr": float(np.mean([row["epona_psnr"] for row in rows])),
        "repeat_psnr": float(np.mean([row["repeat_psnr"] for row in rows])),
        "delta_psnr": float(np.mean([row["delta_psnr"] for row in rows])),
        "epona_ssim": float(np.mean([row["epona_ssim"] for row in rows])),
        "repeat_ssim": float(np.mean([row["repeat_ssim"] for row in rows])),
        "delta_ssim": float(np.mean([row["delta_ssim"] for row in rows])),
        "worst_delta_psnr": float(min(row["delta_psnr"] for row in rows)),
    }
    print(json.dumps(summary, indent=2))
    print()
    for row in rows:
        print(
            f"{row['sample_id']}: dPSNR={row['delta_psnr']:+.4f} "
            f"dSSIM={row['delta_ssim']:+.4f}"
        )


if __name__ == "__main__":
    main()
