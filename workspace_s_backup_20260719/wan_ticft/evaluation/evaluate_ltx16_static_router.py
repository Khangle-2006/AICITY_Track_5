from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.generators import compute_flow, make_region_motion_mask, warp_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_images(cv2, paths: list[str]):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def resize_like(cv2, pred, target):
    if pred.shape[:2] == target.shape[:2]:
        return pred
    return cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)


def read_video_frames(cv2, path: Path, count: int):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def region_predictions(cv2, np, conditions, count: int, history_gap: int, view_type: str):
    last = conditions[-1]
    flow = compute_flow(cv2, np, conditions[-2], last, 426)
    args = argparse.Namespace(
        region_motion_tau=1.5,
        region_motion_temp=0.75,
        region_max_alpha=0.85,
        region_overhead_max_alpha=0.45,
        region_mask_blur=3.0,
    )
    mask, _ = make_region_motion_mask(cv2, np, flow, view_type, args)
    mask = mask[:, :, None]
    last_float = last.astype(np.float32)
    scale = 0.0
    velocity = 1.0 / max(1, history_gap)
    preds = []
    for _ in range(count):
        scale = min(6.0, scale + velocity)
        velocity *= 0.96
        warped = warp_rgb(cv2, np, last, flow, scale).astype(np.float32)
        preds.append(np.clip((1.0 - mask) * last_float + mask * warped, 0, 255).astype(np.uint8))
    return preds


def score(cv2, np, preds, targets):
    values = []
    for pred, target in zip(preds, targets):
        pred = resize_like(cv2, pred, target)
        values.append(
            (
                psnr(cv2, np, pred, target),
                ssim_gray(cv2, np, pred, target),
                mean_absdiff(np, pred, target),
            )
        )
    return tuple(float(np.mean([item[index] for item in values])) for index in range(3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ltx16_report", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--static_fraction", type=float, default=0.20)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows = read_jsonl(args.manifest)
    with args.ltx16_report.open(encoding="utf-8", newline="") as f:
        report = {row["sample_id"]: Path(row["video"]) for row in csv.DictReader(f)}

    candidates = []
    for row in rows:
        conditions = load_images(cv2, (row.get("frames") or {}).get("condition") or [])
        targets = load_images(cv2, (row.get("frames") or {}).get("target") or [])
        if len(conditions) < 2 or not targets or row["sample_id"] not in report:
            continue
        indices = (row.get("frame_indices") or {}).get("condition") or []
        history_gap = max(1, int(indices[-1]) - int(indices[-2])) if len(indices) >= 2 else 1
        motion = mean_absdiff(np, conditions[-2], conditions[-1]) / history_gap
        ltx = read_video_frames(cv2, report[row["sample_id"]], len(targets))
        region = region_predictions(cv2, np, conditions, len(targets), history_gap, row["view_type"])
        candidates.append(
            {
                "sample_id": row["sample_id"],
                "motion": motion,
                "ltx": score(cv2, np, ltx, targets),
                "region": score(cv2, np, region, targets),
            }
        )

    static_count = max(1, round(len(candidates) * args.static_fraction))
    static_ids = {row["sample_id"] for row in sorted(candidates, key=lambda row: row["motion"])[:static_count]}
    output = []
    for row in candidates:
        choice = "region" if row["sample_id"] in static_ids else "ltx16"
        chosen = row["region"] if choice == "region" else row["ltx"]
        output.append(
            {
                "sample_id": row["sample_id"],
                "motion": row["motion"],
                "choice": choice,
                "psnr": chosen[0],
                "ssim": chosen[1],
                "absdiff": chosen[2],
                "ltx_psnr": row["ltx"][0],
                "region_psnr": row["region"][0],
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(f"static route count: {static_count}/{len(output)}")
    print(f"PSNR: {np.mean([row['psnr'] for row in output]):.4f}")
    print(f"SSIM: {np.mean([row['ssim'] for row in output]):.4f}")
    print(f"AbsDiff: {np.mean([row['absdiff'] for row in output]):.4f}")
    print(f"output: {args.out_csv}")


if __name__ == "__main__":
    main()
