"""Materialize actor-protected a100 depth frames for BDD test folders with per-sample resume."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline.frame_io import numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output
from wan_ticft.evaluation.run_sa2va_actor_aware_depth_gate import (
    composite,
    infer_masks_chunked,
    select_track_component,
    soften,
)
from wan_ticft.evaluation.run_sa2va_caption_actor_bdd4 import read_jsonl


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_frames(paths: list[Path]) -> list[np.ndarray]:
    frames = []
    for path in paths:
        frame = read_rgb(cv2, path)
        if frame is None:
            raise RuntimeError(f"cannot read {path}")
        frames.append(frame)
    return frames


def actor_text(prompt: str) -> str:
    match = re.search(r"Pedestrian(?: state)?:\s*(.*?)(?:\s+Vehicle(?: state)?:|$)", prompt, re.DOTALL)
    text = match.group(1) if match else prompt
    return " ".join(text.split())[:900]


def segmentation_prompt(prompt: str) -> str:
    return (
        "<image>The first frame is a real observation and the rest are consecutive generated future frames. "
        "Segment exactly the same primary pedestrian in every frame where visible. Do not segment another "
        "person, vehicles, road, shadows, reflections, or duplicated artifacts. Return no mask after the "
        "pedestrian leaves view. Pedestrian description: "
        + actor_text(prompt)
    )


def mask_centroid(mask: np.ndarray | None):
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean() / mask.shape[1]), float(ys.mean() / mask.shape[0])


def write_tables(out_dir: Path, reports: list[dict], diagnostics: list[dict]) -> None:
    if reports:
        with (out_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
            writer.writeheader()
            writer.writerows(reports)
    if diagnostics:
        with (out_dir / "diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
            writer.writeheader()
            writer.writerows(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--depth_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", default="ByteDance/Sa2VA-4B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--chunk_frames", type=int, default=10)
    parser.add_argument("--dilation", type=int, default=15)
    parser.add_argument("--blur_sigma", type=float, default=3.0)
    parser.add_argument("--scale", type=float, default=4.0 / 3.0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = args.out_dir / ".sa2va_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest = {row["sample_id"]: row for row in read_jsonl(args.manifest)}
    rows = [row for row in read_csv(args.test_csv) if row.get("view_type") == "bdd"]
    if len(rows) != 44:
        raise RuntimeError(f"expected 44 BDD test rows, found {len(rows)}")

    pending = []
    report_rows, diagnostics = [], []
    for row in rows:
        marker = meta_dir / f"{row['source_folder']}.json"
        verified = verify_sample_output(args.out_dir / row["source_folder"], int(row["target_N"]))
        if marker.is_file() and verified.ok:
            cached = json.loads(marker.read_text(encoding="utf-8"))
            report_rows.append(cached["report"])
            diagnostics.extend(cached["diagnostics"])
        else:
            pending.append(row)
    print(f"resume complete={len(report_rows)} pending={len(pending)}", flush=True)

    if pending:
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
    else:
        tokenizer = model = None

    for position, row in enumerate(pending, start=1):
        output_name = row["source_folder"]
        sample_id = row["sample_id"]
        target_n = int(row["target_N"])
        manifest_row = manifest.get(sample_id)
        if manifest_row is None:
            raise RuntimeError(f"missing manifest row: {sample_id}")
        conditions = load_frames(numeric_pngs(Path(row["input_dir"])))
        bases = load_frames(numeric_pngs(args.base_dir / output_name))
        depths = load_frames(numeric_pngs(args.depth_dir / output_name))
        if not conditions or len(bases) != target_n or len(depths) != target_n:
            raise RuntimeError(
                f"{output_name}: conditions={len(conditions)} base={len(bases)} depth={len(depths)} expected={target_n}"
            )
        height, width = bases[0].shape[:2]
        bases = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in bases]
        depths = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in depths]
        anchor = conditions[-1]
        print(f"[{position}/{len(pending)}] {sample_id} frames={target_n}", flush=True)
        anchor_mask, raw_masks = infer_masks_chunked(
            model,
            tokenizer,
            anchor,
            bases,
            segmentation_prompt(str(manifest_row.get("prompt", ""))),
            args.chunk_frames,
        )
        previous = mask_centroid(anchor_mask)
        selected_masks, sample_diagnostics = [], []
        for frame_index, (raw, frame) in enumerate(zip(raw_masks, bases)):
            if raw is None:
                raw = np.zeros(frame.shape[:2], dtype=bool)
            selected, next_point, components = select_track_component(raw, previous)
            if selected.shape != frame.shape[:2]:
                selected = np.zeros(frame.shape[:2], dtype=bool)
            if selected.any():
                previous = next_point
            selected_masks.append(selected)
            sample_diagnostics.append(
                {
                    "sample_id": sample_id,
                    "output_name": output_name,
                    "frame": frame_index,
                    "raw_nonempty": bool(raw.any()),
                    "components": components,
                    "selected_area_fraction": float(selected.mean()),
                    "centroid_x": "" if previous is None else previous[0],
                    "centroid_y": "" if previous is None else previous[1],
                }
            )
        sample_out = args.out_dir / output_name
        if sample_out.exists():
            shutil.rmtree(sample_out)
        sample_out.mkdir(parents=True)
        for frame_index, (base, depth, mask) in enumerate(zip(bases, depths, selected_masks)):
            output = composite(base, depth, soften(mask, args.dilation, args.blur_sigma), args.scale)
            if not write_rgb(cv2, sample_out / f"{frame_index}.png", output):
                raise RuntimeError(f"cannot write {sample_out / f'{frame_index}.png'}")
        verified = verify_sample_output(sample_out, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verification failed: {verified.notes}")
        report = {
            "output_name": output_name,
            "sample_id": sample_id,
            "view_type": "bdd",
            "target_N": target_n,
            "frames_found": verified.frames_found,
            "scale_from_a75": f"{args.scale:.6f}",
            "status": "ok",
        }
        marker = {"report": report, "diagnostics": sample_diagnostics}
        (meta_dir / f"{output_name}.json").write_text(json.dumps(marker), encoding="utf-8")
        report_rows.append(report)
        diagnostics.extend(sample_diagnostics)
        write_tables(args.out_dir, report_rows, diagnostics)
        print(f"[{position}/{len(pending)}] {output_name} | ok", flush=True)
        torch.cuda.empty_cache()

    report_rows.sort(key=lambda item: item["output_name"])
    diagnostics.sort(key=lambda item: (item["output_name"], item["frame"]))
    write_tables(args.out_dir, report_rows, diagnostics)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
