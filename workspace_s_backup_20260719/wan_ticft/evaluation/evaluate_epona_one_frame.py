#!/usr/bin/env python3
import argparse
import json

import cv2
import numpy as np


def ssim_gray(reference, prediction):
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float64)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_ref = cv2.GaussianBlur(reference, (11, 11), 1.5)
    mu_pred = cv2.GaussianBlur(prediction, (11, 11), 1.5)
    mu_ref_sq = mu_ref * mu_ref
    mu_pred_sq = mu_pred * mu_pred
    mu_cross = mu_ref * mu_pred
    sigma_ref = cv2.GaussianBlur(reference * reference, (11, 11), 1.5) - mu_ref_sq
    sigma_pred = cv2.GaussianBlur(prediction * prediction, (11, 11), 1.5) - mu_pred_sq
    sigma_cross = cv2.GaussianBlur(reference * prediction, (11, 11), 1.5) - mu_cross
    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_ref_sq + mu_pred_sq + c1) * (sigma_ref + sigma_pred + c2)
    return float(np.mean(numerator / denominator))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument(
        "--prediction_rotation",
        choices=["none", "clockwise", "counterclockwise"],
        default="none",
    )
    args = parser.parse_args()

    pred = cv2.imread(args.prediction)
    baseline = cv2.imread(args.baseline)
    ref = cv2.imread(args.reference)
    if pred is None or baseline is None or ref is None:
        raise RuntimeError("prediction, baseline, or reference image is missing")
    if args.prediction_rotation == "clockwise":
        pred = cv2.rotate(pred, cv2.ROTATE_90_CLOCKWISE)
    elif args.prediction_rotation == "counterclockwise":
        pred = cv2.rotate(pred, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def metrics(image):
        image = cv2.resize(image, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
        diff = image.astype(np.float32) - ref.astype(np.float32)
        mse = float(np.mean(diff * diff))
        return {
            "psnr": float(10.0 * np.log10((255.0 * 255.0) / max(mse, 1e-12))),
            "ssim": ssim_gray(ref, image),
            "absdiff": float(np.mean(np.abs(diff))),
        }

    epona = metrics(pred)
    repeat_last = metrics(baseline)
    result = {
        "epona": epona,
        "repeat_last": repeat_last,
        "delta_psnr": epona["psnr"] - repeat_last["psnr"],
        "delta_ssim": epona["ssim"] - repeat_last["ssim"],
        "reference_shape": list(ref.shape),
    }
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
