"""Track a caption-referred actor from the last observation into sampled CNN/LTX predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from wan_ticft.evaluation.run_sa2va_caption_actor_bdd4 import (
    as_video_mask,
    mask_stats,
    overlay_mask,
    read_jsonl,
    resize_mask,
    tile,
)


def read_report(path: Path) -> dict[str, Path]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["sample_id"]: Path(row["video"])
            for row in csv.DictReader(handle)
            if row.get("status") == "ok" and row.get("video")
        }


def actor_description(prompt: str) -> str:
    start = prompt.find("Pedestrian state:")
    end = prompt.find("Vehicle state:", start + 1)
    if start >= 0:
        prompt = prompt[start + len("Pedestrian state:") : end if end >= 0 else None]
    return " ".join(prompt.split())[:900]


def future_prompt(row: dict) -> str:
    return (
        "<image>The first frame is a real observation. The remaining frames are sampled from a generated "
        "near-future video. Segment exactly the same primary pedestrian in every frame where that person "
        "is visible. Do not segment another pedestrian, vehicles, road, buildings, shadows, reflections, "
        "or duplicated artifacts. A missing mask is correct after the person leaves the camera view. "
        f"Pedestrian description: {actor_description(str(row.get('prompt', '')))}"
    )


def read_video_samples(cv2, path: Path, count: int) -> tuple[list[np.ndarray], list[int], int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        raise RuntimeError(f"video has no frames: {path}")
    indices = np.linspace(0, total - 1, min(count, total)).round().astype(int).tolist()
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            cap.release()
            raise RuntimeError(f"cannot read {path} at frame {index}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, indices, total


def make_sheet(cv2, frames, overlays, labels, sample_id: str):
    panels = [tile(cv2, frame, label, size=(320, 180)) for frame, label in zip(frames, labels)]
    masked = [tile(cv2, frame, f"mask {label}", size=(320, 180)) for frame, label in zip(overlays, labels)]
    blank = np.zeros_like(panels[0])
    while len(panels) % 5:
        panels.append(blank.copy())
        masked.append(blank.copy())
    original_rows = [np.concatenate(panels[i : i + 5], axis=1) for i in range(0, len(panels), 5)]
    mask_rows = [np.concatenate(masked[i : i + 5], axis=1) for i in range(0, len(masked), 5)]
    sheet = np.concatenate(original_rows + mask_rows, axis=0)
    cv2.putText(sheet, sample_id, (8, sheet.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", default="ByteDance/Sa2VA-4B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--sample_id", action="append", default=[])
    parser.add_argument("--future_samples", type=int, default=9)
    args = parser.parse_args()

    import cv2
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    order = {sample_id: index for index, sample_id in enumerate(args.sample_id)}
    rows = [row for row in read_jsonl(args.manifest) if row.get("sample_id") in order]
    rows.sort(key=lambda row: order[row["sample_id"]])
    if len(rows) != len(args.sample_id):
        raise RuntimeError(f"requested {len(args.sample_id)} rows but found {len(rows)}")
    videos = read_report(args.base_report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    report = []
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        video_path = videos.get(sample_id)
        if video_path is None or not video_path.is_file():
            raise RuntimeError(f"missing cnn_ltxmix65 video for {sample_id}: {video_path}")
        future_frames, future_indices, total = read_video_samples(cv2, video_path, args.future_samples)
        condition_paths = (row.get("frames") or {}).get("condition") or []
        if not condition_paths:
            raise RuntimeError(f"missing condition frames for {sample_id}")
        anchor = np.asarray(Image.open(condition_paths[-1]).convert("RGB"))
        frames = [anchor, *future_frames]
        labels = ["real anchor", *[f"future {index}/{total - 1}" for index in future_indices]]
        prompt = future_prompt(row)
        result = model.predict_forward(
            video=[Image.fromarray(frame) for frame in frames],
            text=prompt,
            past_text="",
            mask_prompts=None,
            tokenizer=tokenizer,
        )
        prediction = str(result.get("prediction", ""))
        video_mask = as_video_mask(result.get("prediction_masks"), len(frames))
        sample_dir = args.out_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        (sample_dir / "prediction.txt").write_text(prediction + "\n", encoding="utf-8")
        if video_mask is None:
            masks = [np.zeros(frame.shape[:2], dtype=bool) for frame in frames]
            status = "no_video_mask"
        else:
            masks = [resize_mask(cv2, mask, frame.shape[1], frame.shape[0]) for mask, frame in zip(video_mask, frames)]
            np.savez_compressed(
                sample_dir / "sampled_masks.npz",
                **{f"mask_{index:02d}": mask for index, mask in enumerate(masks)},
            )
            status = "ok" if masks[0].any() else "missing_anchor"
        overlays = [overlay_mask(cv2, frame, mask) for frame, mask in zip(frames, masks)]
        Image.fromarray(make_sheet(cv2, frames, overlays, labels, sample_id)).save(
            sample_dir / "future_segmentation_sheet.jpg", quality=94
        )
        stats = [mask_stats(mask) for mask in masks]
        report.append(
            {
                "sample_id": sample_id,
                "status": status,
                "total_video_frames": total,
                "sampled_indices": json.dumps(future_indices),
                "anchor_nonempty": bool(masks[0].any()),
                "future_nonempty": sum(mask.any() for mask in masks[1:]),
                "mean_future_area_fraction": float(np.mean([item[0] for item in stats[1:]])),
                "centroids_x": json.dumps([item[1] for item in stats]),
                "centroids_y": json.dumps([item[2] for item in stats]),
                "prediction": prediction.replace("\n", " ")[:500],
                "sheet": str(sample_dir / "future_segmentation_sheet.jpg"),
            }
        )
        print(
            f"[{position}/{len(rows)}] {sample_id} | {status} "
            f"anchor={int(masks[0].any())} future={report[-1]['future_nonempty']}/{len(masks) - 1}",
            flush=True,
        )

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print(f"report={report_path}")
    print(f"sheets={args.out_dir}/*/future_segmentation_sheet.jpg")


if __name__ == "__main__":
    main()
