"""Audit caption-referred pedestrian masks from Sa2VA on four BDD histories."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def actor_description(prompt: str) -> str:
    match = re.search(r"Pedestrian state:\s*(.*?)(?:\s+Vehicle state:|$)", prompt, re.DOTALL)
    text = match.group(1) if match else prompt
    text = re.sub(r"\s+", " ", text).strip()
    return text[:900]


def segmentation_prompt(row: dict) -> str:
    description = actor_description(str(row.get("prompt", "")))
    return (
        "<image>Please segment exactly one object: the primary pedestrian described below. "
        "Track that same physical person through every video frame. Do not segment vehicles, "
        "road, buildings, shadows, reflections, or multiple past positions of the person. "
        "If the described pedestrian is not visible, return no segmentation mask. "
        f"Pedestrian description: {description}"
    )


def as_video_mask(prediction_masks, frame_count: int) -> np.ndarray | None:
    if prediction_masks is None:
        return None
    candidates = prediction_masks if isinstance(prediction_masks, (list, tuple)) else [prediction_masks]
    for candidate in candidates:
        if hasattr(candidate, "detach"):
            candidate = candidate.detach().float().cpu().numpy()
        array = np.asarray(candidate)
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 2 and frame_count == 1:
            array = array[None]
        if array.ndim == 3 and array.shape[0] == frame_count:
            return array > 0
    return None


def resize_mask(cv2, mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def overlay_mask(cv2, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = rgb.copy()
    tint = np.zeros_like(output)
    tint[..., 0] = 255
    output[mask] = (0.58 * output[mask] + 0.42 * tint[mask]).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 255, 80), 2)
    return output


def tile(cv2, rgb: np.ndarray, label: str, size=(384, 216)) -> np.ndarray:
    panel = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (size[0], 27), (0, 0, 0), -1)
    cv2.putText(panel, label, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def make_sheet(cv2, frames: list[np.ndarray], overlays: list[np.ndarray], sample_id: str) -> np.ndarray:
    originals = [tile(cv2, frame, f"history {index}") for index, frame in enumerate(frames)]
    masks = [tile(cv2, frame, f"Sa2VA actor mask {index}") for index, frame in enumerate(overlays)]
    sheet = np.concatenate([np.concatenate(originals, axis=1), np.concatenate(masks, axis=1)], axis=0)
    cv2.putText(sheet, sample_id, (8, sheet.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


def mask_stats(mask: np.ndarray) -> tuple[float, float, float]:
    area = float(mask.mean())
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return area, float("nan"), float("nan")
    return area, float(xs.mean() / mask.shape[1]), float(ys.mean() / mask.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", default="ByteDance/Sa2VA-4B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--sample_id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    import cv2
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"]
    if args.sample_id:
        order = {sample_id: index for index, sample_id in enumerate(args.sample_id)}
        rows = [row for row in rows if row.get("sample_id") in order]
        rows.sort(key=lambda row: order[row["sample_id"]])
    rows = rows[: args.limit]
    if len(rows) != args.limit:
        raise RuntimeError(f"expected {args.limit} BDD rows, found {len(rows)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        use_fast=False,
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
        paths = (row.get("frames") or {}).get("condition") or []
        frames = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
        if len(frames) != 4:
            raise RuntimeError(f"{sample_id}: expected 4 condition frames, found {len(frames)}")
        prompt = segmentation_prompt(row)
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
            np.save(sample_dir / "masks.npy", np.stack(masks))
            status = "ok" if all(mask.any() for mask in masks) else "partial"
        overlays = [overlay_mask(cv2, frame, mask) for frame, mask in zip(frames, masks)]
        sheet = make_sheet(cv2, frames, overlays, sample_id)
        Image.fromarray(sheet).save(sample_dir / "segmentation_sheet.jpg", quality=94)
        for frame_index, overlay in enumerate(overlays):
            Image.fromarray(overlay).save(sample_dir / f"overlay_{frame_index:02d}.png")

        stats = [mask_stats(mask) for mask in masks]
        nonempty = sum(area > 0 for area, _, _ in stats)
        nonzero_areas = [area for area, _, _ in stats if area > 0]
        area_ratio = max(nonzero_areas) / min(nonzero_areas) if nonzero_areas else float("inf")
        report.append(
            {
                "sample_id": sample_id,
                "status": status,
                "nonempty_frames": nonempty,
                "mean_area_fraction": float(np.mean([item[0] for item in stats])),
                "max_min_area_ratio": area_ratio,
                "centroids_x": json.dumps([item[1] for item in stats]),
                "centroids_y": json.dumps([item[2] for item in stats]),
                "prediction": prediction.replace("\n", " ")[:500],
                "sheet": str(sample_dir / "segmentation_sheet.jpg"),
            }
        )
        print(f"[{position}/{len(rows)}] {sample_id} | {status} nonempty={nonempty}/4 area={report[-1]['mean_area_fraction']:.5f}")

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print(f"report={report_path}")
    print(f"sheets={args.out_dir}/*/segmentation_sheet.jpg")


if __name__ == "__main__":
    main()
