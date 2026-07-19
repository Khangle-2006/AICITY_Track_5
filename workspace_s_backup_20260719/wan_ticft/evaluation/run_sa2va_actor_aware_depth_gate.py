"""Protect a Sa2VA-tracked pedestrian while strengthening depth-rendered background texture."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from wan_ticft.evaluation.run_sa2va_caption_actor_bdd4 import as_video_mask, read_jsonl, resize_mask
from wan_ticft.evaluation.run_sa2va_future_actor_gate import actor_description


def read_report(path: Path) -> dict[str, Path]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["sample_id"]: Path(row["video"])
            for row in csv.DictReader(handle)
            if row.get("status") == "ok" and row.get("video")
        }


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"video has no frames: {path}")
    return frames, fps


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def prompt_for(row: dict) -> str:
    return (
        "<image>The first frame is a real observation and the rest are consecutive generated future frames. "
        "Segment exactly the same primary pedestrian in every frame where visible. Do not segment another "
        "person, vehicles, road, shadows, reflections, or duplicated artifacts. Return no mask after the "
        "pedestrian leaves view. Pedestrian description: "
        + actor_description(str(row.get("prompt", "")))
    )


def infer_masks_chunked(model, tokenizer, anchor: np.ndarray, frames: list[np.ndarray], prompt: str, chunk: int):
    masks: list[np.ndarray | None] = []
    anchor_mask = None
    for start in range(0, len(frames), chunk):
        batch = frames[start : start + chunk]
        inputs = [anchor, *batch]
        with torch.inference_mode():
            result = model.predict_forward(
                video=[Image.fromarray(frame) for frame in inputs],
                text=prompt,
                past_text="",
                mask_prompts=None,
                tokenizer=tokenizer,
            )
        video_mask = as_video_mask(result.get("prediction_masks"), len(inputs))
        if video_mask is None:
            masks.extend([None] * len(batch))
            del result
            torch.cuda.empty_cache()
            continue
        if anchor_mask is None:
            anchor_mask = resize_mask(cv2, video_mask[0], anchor.shape[1], anchor.shape[0])
        masks.extend([resize_mask(cv2, mask, frame.shape[1], frame.shape[0]) for mask, frame in zip(video_mask[1:], batch)])
        del result, video_mask
        torch.cuda.empty_cache()
        print(f"  masks {start + 1}-{start + len(batch)}/{len(frames)}", flush=True)
    return anchor_mask, masks


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return float(xs.mean() / mask.shape[1]), float(ys.mean() / mask.shape[0])


def select_track_component(mask: np.ndarray | None, previous: tuple[float, float] | None):
    if mask is None or not mask.any():
        return np.zeros(mask.shape if mask is not None else (1, 1), dtype=bool), previous, 0
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    candidates = []
    frame_area = mask.shape[0] * mask.shape[1]
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(4, frame_area * 0.00001):
            continue
        point = (float(centers[label][0] / mask.shape[1]), float(centers[label][1] / mask.shape[0]))
        distance = 0.0 if previous is None else float(np.hypot(point[0] - previous[0], point[1] - previous[1]))
        # Temporal proximity dominates; area only breaks ties and rejects tiny speckles.
        score = distance - 0.015 * np.log1p(area)
        candidates.append((score, label, point, area))
    if not candidates:
        return np.zeros_like(mask, dtype=bool), previous, 0
    _, label, point, _ = min(candidates, key=lambda item: item[0])
    selected = labels == label
    return selected, point, len(candidates)


def soften(mask: np.ndarray, dilation: int, sigma: float) -> np.ndarray:
    kernel = np.ones((dilation, dilation), np.uint8)
    expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return cv2.GaussianBlur(expanded.astype(np.float32), (0, 0), sigma).clip(0.0, 1.0)


def composite(base: np.ndarray, depth_a75: np.ndarray, actor: np.ndarray, scale: float) -> np.ndarray:
    stronger = base.astype(np.float32) + scale * (depth_a75.astype(np.float32) - base.astype(np.float32))
    stronger = np.clip(stronger, 0, 255)
    weight = actor[..., None]
    return np.clip(stronger * (1.0 - weight) + base.astype(np.float32) * weight, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--depth_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", default="ByteDance/Sa2VA-4B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--sample_id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk_frames", type=int, default=3)
    parser.add_argument("--dilation", type=int, default=15)
    parser.add_argument("--blur_sigma", type=float, default=3.0)
    args = parser.parse_args()

    manifest_rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"]
    if args.sample_id:
        order = {sample_id: index for index, sample_id in enumerate(args.sample_id)}
        rows = [row for row in manifest_rows if row.get("sample_id") in order]
        rows.sort(key=lambda row: order[row["sample_id"]])
        if len(rows) != len(args.sample_id):
            raise RuntimeError(f"requested {len(args.sample_id)} rows but found {len(rows)}")
    else:
        rows = manifest_rows[: args.limit or None]
        if args.limit and len(rows) != args.limit:
            raise RuntimeError(f"requested limit={args.limit} but found {len(rows)} BDD rows")
    bases = read_report(args.base_report)
    depths = read_report(args.depth_report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "selected_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    variants = {"actor_depth_a75": 1.0, "actor_depth_a90": 1.20, "actor_depth_a100": 4.0 / 3.0}
    reports = {name: [] for name in variants}
    diagnostics = []
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        if sample_id not in bases or sample_id not in depths:
            raise RuntimeError(f"missing base/depth report row for {sample_id}")
        base_frames, fps = read_video(bases[sample_id])
        depth_frames, _ = read_video(depths[sample_id])
        if len(base_frames) != len(depth_frames):
            raise RuntimeError(f"{sample_id}: base={len(base_frames)} depth={len(depth_frames)} frames")
        height, width = base_frames[0].shape[:2]
        depth_frames = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in depth_frames]
        condition_paths = (row.get("frames") or {}).get("condition") or []
        anchor = np.asarray(Image.open(condition_paths[-1]).convert("RGB"))
        print(f"[{position}/{len(rows)}] {sample_id}", flush=True)
        anchor_raw, raw_masks = infer_masks_chunked(
            model, tokenizer, anchor, base_frames, prompt_for(row), args.chunk_frames
        )
        previous = centroid(anchor_raw) if anchor_raw is not None else None
        selected_masks = []
        for frame_index, (raw, frame) in enumerate(zip(raw_masks, base_frames)):
            if raw is None:
                raw = np.zeros(frame.shape[:2], dtype=bool)
            selected, next_point, component_count = select_track_component(raw, previous)
            if selected.shape != frame.shape[:2]:
                selected = np.zeros(frame.shape[:2], dtype=bool)
            if selected.any():
                previous = next_point
            selected_masks.append(selected)
            diagnostics.append(
                {
                    "sample_id": sample_id,
                    "frame": frame_index,
                    "raw_nonempty": bool(raw.any()),
                    "components": component_count,
                    "selected_area_fraction": float(selected.mean()),
                    "centroid_x": "" if previous is None else previous[0],
                    "centroid_y": "" if previous is None else previous[1],
                }
            )
        np.savez_compressed(
            args.out_dir / f"{sample_id}_masks.npz",
            **{f"mask_{index:03d}": mask for index, mask in enumerate(selected_masks)},
        )
        soft_masks = [soften(mask, args.dilation, args.blur_sigma) for mask in selected_masks]
        for name, scale in variants.items():
            outputs = [
                composite(base, depth, mask, scale)
                for base, depth, mask in zip(base_frames, depth_frames, soft_masks)
            ]
            path = args.out_dir / name / "videos" / f"{sample_id}.avi"
            write_video(path, outputs, fps)
            reports[name].append(
                {"sample_id": sample_id, "video": str(path), "status": "ok", "future_frames": len(outputs)}
            )

    for name, rows_out in reports.items():
        path = args.out_dir / name / "report.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
    with (args.out_dir / "diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
