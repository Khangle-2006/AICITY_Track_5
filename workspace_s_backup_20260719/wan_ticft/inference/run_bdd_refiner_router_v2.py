from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from baseline.console import kv, section
from baseline.frame_io import import_cv2_np, load_history_frames, write_rgb
from baseline.index_io import load_test_rows, write_manifest
from baseline.naming import parse_int, resolve_output_name
from baseline.verify import verify_sample_output
from wan_ticft.evaluation.evaluate_bdd_ego_motion import (
    combine_predictions,
    load_refiner,
    predict,
    refiner_predictions,
)


PHASE_ROUTE = {
    "0": "refiner_blend_25",
    "1": "refiner_blend_50",
    "2": "refiner_blend_25",
    "3": "refiner_blend_50",
    "4": "refiner_blend_25",
}


def process_row(cv2, np, refiner, row: dict[str, str], output_name: str, args) -> dict[str, Any]:
    target_n = parse_int(row.get("target_N", ""), 0)
    width = parse_int(row.get("frame_width", ""), 1280)
    height = parse_int(row.get("frame_height", ""), 720)
    out_dir = Path(args.out) / output_name
    phase = str(row.get("phase", ""))
    route = PHASE_ROUTE.get(phase, "refiner_blend_50")

    result = {
        "sample_id": row.get("sample_id", ""),
        "source_folder": row.get("source_folder", ""),
        "dataset_part": row.get("dataset_part", ""),
        "view_type": row.get("view_type", ""),
        "phase": phase,
        "history_K": row.get("history_K", ""),
        "target_N": target_n,
        "frame_width": width,
        "frame_height": height,
        "input_dir": row.get("input_dir", ""),
        "output_dir": str(out_dir),
        "method": f"bdd_refiner_router_v2:{route}",
        "frames_written": 0,
        "status": "pending",
        "notes": "",
    }

    frames, frame_paths = load_history_frames(cv2, row, width, height)
    if len(frames) < 2 or target_n <= 0:
        result["status"] = "failed"
        result["notes"] = "requires at least two history frames and positive target_N"
        return result

    history_gap = 1
    if len(frame_paths) >= 2:
        history_gap = max(1, int(frame_paths[-1].stem) - int(frame_paths[-2].stem))
    offsets = list(range(1, target_n + 1))
    classical, _, _ = predict(
        cv2,
        np,
        frames,
        offsets,
        history_gap,
        {"kind": "phase_router"},
        args.estimate_width,
        phase,
    )
    learned = refiner_predictions(cv2, np, refiner, frames, offsets, history_gap, phase)
    predictions = combine_predictions(np, classical, learned, offsets, route)

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = sum(write_rgb(cv2, out_dir / f"{i}.png", pred) for i, pred in enumerate(predictions))
    verified = verify_sample_output(out_dir, target_n)
    result["frames_written"] = written
    result["status"] = "ok" if verified.ok and written == target_n else "bad"
    result["notes"] = f"route={route}; history_gap={history_gap}; {verified.notes}".rstrip("; ")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=Path, default=Path("/workspace/experiments/index"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--estimate_width", type=int, default=640)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    refiner = load_refiner(args.checkpoint)
    all_rows = load_test_rows(args.index_dir)
    used: set[str] = set()
    named_rows = [(row, resolve_output_name(row, used)) for row in all_rows]
    named_rows = [(row, name) for row, name in named_rows if row.get("view_type") == "bdd"]
    if args.limit > 0:
        named_rows = named_rows[: args.limit]

    section("BDD REFINER ROUTER V2")
    kv("checkpoint", args.checkpoint)
    kv("out", args.out)
    kv("rows", len(named_rows))
    kv("phase routes", PHASE_ROUTE)

    outputs = []
    for index, (row, output_name) in enumerate(named_rows, start=1):
        print(f"[{index}/{len(named_rows)}] {output_name}", flush=True)
        outputs.append(process_row(cv2, np, refiner, row, output_name, args))

    manifest = args.out / "manifest.csv"
    write_manifest(manifest, outputs)
    summary = {
        "rows": len(outputs),
        "frames_written": sum(int(row["frames_written"]) for row in outputs),
        "by_status": {
            status: sum(row["status"] == status for row in outputs)
            for status in sorted({str(row["status"]) for row in outputs})
        },
        "phase_routes": PHASE_ROUTE,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    section("RESULT")
    kv("manifest", manifest)
    kv("summary", args.out / "summary.json")
    kv("rows", summary["rows"])
    kv("frames_written", summary["frames_written"])
    kv("by_status", summary["by_status"])
    if any(row["status"] != "ok" for row in outputs):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
