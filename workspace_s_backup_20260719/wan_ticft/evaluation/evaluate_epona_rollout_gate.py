#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_epona_one_frame_gate import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--reference_root", required=True)
    parser.add_argument("--prediction_root", required=True)
    parser.add_argument("--future_frames", type=int, default=4)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    rows = []
    sample_dirs = sorted(path for path in Path(args.data_root).iterdir() if path.is_dir())
    for index, sample_dir in enumerate(sample_dirs):
        sample_id = sample_dir.name
        baseline = cv2.imread(str(sample_dir / "000009.png"))
        for step in range(1, args.future_frames + 1):
            pred = cv2.imread(
                str(Path(args.prediction_root) / f"sliding_{index}" / f"{9 + step}.png")
            )
            reference = cv2.imread(
                str(Path(args.reference_root) / sample_id / f"future_{step:03d}.png")
            )
            if pred is None or baseline is None or reference is None:
                raise RuntimeError(f"missing image for {sample_id}, step {step}")
            epona = metrics(pred, reference)
            repeat = metrics(baseline, reference)
            rows.append(
                {
                    "sample_id": sample_id,
                    "step": step,
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
        "rows": len(rows),
        "wins": sum(row["delta_psnr"] > 0 for row in rows),
        "delta_psnr": float(np.mean([row["delta_psnr"] for row in rows])),
        "delta_ssim": float(np.mean([row["delta_ssim"] for row in rows])),
        "worst_delta_psnr": float(min(row["delta_psnr"] for row in rows)),
        "step4_delta_psnr": float(
            np.mean([row["delta_psnr"] for row in rows if row["step"] == args.future_frames])
        ),
        "step4_delta_ssim": float(
            np.mean([row["delta_ssim"] for row in rows if row["step"] == args.future_frames])
        ),
    }
    print(json.dumps(summary, indent=2))
    print()
    for row in rows:
        print(
            f"{row['sample_id']} step{row['step']}: "
            f"dPSNR={row['delta_psnr']:+.4f} dSSIM={row['delta_ssim']:+.4f}"
        )


if __name__ == "__main__":
    main()
