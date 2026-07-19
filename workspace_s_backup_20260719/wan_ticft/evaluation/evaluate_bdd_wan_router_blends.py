from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baseline.frame_io import import_cv2_np
from wan_ticft.evaluation.evaluate_bdd_ego_motion import (
    combine_predictions,
    load_images,
    load_refiner,
    predict,
    refiner_predictions,
    score_sequence,
    target_offsets,
)
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_manifest,
    read_report,
    read_video_rgb_frames,
    resize_like,
    video_fps,
)


PHASE_ROUTE = {
    "0": "refiner_blend_25",
    "1": "refiner_blend_50",
    "2": "refiner_blend_25",
    "3": "refiner_blend_50",
    "4": "refiner_blend_25",
}

BLENDS = (10, 20, 30, 40, 50)
FIELDS = ("sample_id", "phase", "candidate", "psnr", "ssim", "absdiff", "status", "notes")


def blend(np, wan, router, alpha: float):
    return [
        np.clip(
            (1.0 - alpha) * wan_frame.astype(np.float32)
            + alpha * router_frame.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)
        for wan_frame, router_frame in zip(wan, router)
    ]


def print_summary(np, rows):
    labels = sorted({row["candidate"] for row in rows})
    print(f"{'Candidate':<28} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10} {'Wins':>7}")
    print("-" * 76)
    winners = {}
    samples = {}
    for row in rows:
        if row["status"] == "ok":
            samples.setdefault(row["sample_id"], []).append(row)
    for items in samples.values():
        winner = max(items, key=lambda item: float(item["psnr"]))
        winners[winner["candidate"]] = winners.get(winner["candidate"], 0) + 1
    for label in labels:
        items = [row for row in rows if row["candidate"] == label and row["status"] == "ok"]
        if items:
            print(
                f"{label:<28} {len(items):>5} "
                f"{np.mean([float(x['psnr']) for x in items]):>10.4f} "
                f"{np.mean([float(x['ssim']) for x in items]):>10.4f} "
                f"{np.mean([float(x['absdiff']) for x in items]):>10.4f} "
                f"{winners.get(label, 0):>7}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wan_report", type=Path, required=True)
    parser.add_argument("--refiner_checkpoint", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--estimate_width", type=int, default=640)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    refiner = load_refiner(args.refiner_checkpoint)
    wan_videos = read_report(args.wan_report)
    manifest = [row for row in read_manifest(args.manifest) if row.get("view_type") == "bdd"]
    rows = []

    for index, row in enumerate(manifest, start=1):
        sample_id = str(row.get("sample_id", ""))
        phase = str(row.get("phase", ""))
        print(f"[{index}/{len(manifest)}] {sample_id}", flush=True)
        conditions = load_images(cv2, (row.get("frames") or {}).get("condition") or [])
        targets = load_images(cv2, (row.get("frames") or {}).get("target") or [])
        indices = (row.get("frame_indices") or {}).get("condition") or []
        history_gap = max(1, int(indices[-1]) - int(indices[-2])) if len(indices) >= 2 else 1
        offsets = target_offsets(row, len(targets))
        video = wan_videos.get(sample_id)

        if not conditions or not targets or video is None:
            rows.append(
                {
                    "sample_id": sample_id,
                    "phase": phase,
                    "candidate": "wan",
                    "status": "bad",
                    "notes": "missing condition, target, or Wan video",
                }
            )
            continue

        classical, _, _ = predict(
            cv2,
            np,
            conditions,
            offsets,
            history_gap,
            {"kind": "phase_router"},
            args.estimate_width,
            phase,
        )
        learned = refiner_predictions(cv2, np, refiner, conditions, offsets, history_gap, phase)
        router = combine_predictions(np, classical, learned, offsets, PHASE_ROUTE.get(phase, "refiner_blend_25"))

        gen_offsets = map_target_offsets(row, video_fps(cv2, video))
        frame_map = read_video_rgb_frames(cv2, video, gen_offsets)
        wan = [resize_like(cv2, frame_map[offset], target) for offset, target in zip(gen_offsets, targets) if offset in frame_map]
        candidates = {"wan": wan, "sharp_router": router}
        if len(wan) == len(targets):
            for percent in BLENDS:
                candidates[f"wan_router_{percent}"] = blend(np, wan, router, percent / 100.0)

        for label, predictions in candidates.items():
            status = "ok" if len(predictions) == len(targets) else "bad"
            p, s, d = score_sequence(cv2, np, predictions, targets) if status == "ok" else (0.0, 0.0, 0.0)
            rows.append(
                {
                    "sample_id": sample_id,
                    "phase": phase,
                    "candidate": label,
                    "psnr": f"{p:.6f}" if status == "ok" else "",
                    "ssim": f"{s:.6f}" if status == "ok" else "",
                    "absdiff": f"{d:.6f}" if status == "ok" else "",
                    "status": status,
                    "notes": "",
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)
    print_summary(np, rows)
    print(f"output {args.out_csv}")


if __name__ == "__main__":
    main()

