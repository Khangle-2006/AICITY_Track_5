from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .console import kv, section
from .frame_io import clean_output_dir, import_cv2_np, load_history_frames
from .generators import (
    copy_or_write_repeat,
    generate_flow_decay,
    generate_flow_light,
    generate_actor_layer_small_filled,
    generate_motion_adaptive,
    generate_region_flow_blend,
    generate_region_norm_medium,
)
from .index_io import load_test_rows, write_manifest
from .montage import make_montage
from .naming import parse_int, resolve_output_name
from .verify import verify_sample_output


def process_row(cv2, np, row: dict[str, str], output_name: str, args) -> dict[str, Any]:
    target_n = parse_int(row.get("target_N", ""), 0)
    width = parse_int(row.get("frame_width", ""), 1280)
    height = parse_int(row.get("frame_height", ""), 720)
    out_dir = Path(args.out) / output_name

    base = {
        "sample_id": row.get("sample_id", ""),
        "source_folder": row.get("source_folder", ""),
        "dataset_part": row.get("dataset_part", ""),
        "view_type": row.get("view_type", ""),
        "phase": row.get("phase", ""),
        "history_K": row.get("history_K", ""),
        "target_N": target_n,
        "frame_width": width,
        "frame_height": height,
        "input_dir": row.get("input_dir", ""),
        "output_dir": str(out_dir),
        "method": args.method,
        "frames_written": 0,
        "status": "pending",
        "notes": "",
    }

    if args.verify_only:
        vr = verify_sample_output(out_dir, target_n)
        base["frames_written"] = vr.frames_found
        base["status"] = "ok" if vr.ok else "bad"
        base["notes"] = vr.notes
        return base

    if args.dry_run:
        base["status"] = "dry_run"
        base["notes"] = "not generated"
        return base

    if target_n <= 0:
        base["status"] = "failed"
        base["notes"] = "invalid target_N"
        return base

    frames, frame_paths = load_history_frames(cv2, row, width, height)
    if not frames:
        base["status"] = "failed"
        base["notes"] = f"no input frames found in {row.get('input_dir', '')}"
        return base

    if args.clean:
        clean_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, float] = {}
    method = args.method
    if method == "view_hybrid":
        method = "flow_light" if row.get("view_type", "") == "wts_overhead_or_fixed_camera" else "flow_decay"

    if method == "repeat" or len(frames) < 2:
        written = copy_or_write_repeat(cv2, frames[-1], out_dir, target_n, args.png_compression)
        if method != "repeat" and len(frames) < 2:
            base["notes"] = "fallback_repeat_less_than_2_history_frames"
    elif method == "flow_decay":
        written, stats = generate_flow_decay(cv2, np, frames, out_dir, target_n, args)
    elif method == "flow_light":
        written, stats = generate_flow_light(cv2, np, frames, out_dir, target_n, args)
    elif method == "motion_adaptive":
        written, stats = generate_motion_adaptive(cv2, np, frames, out_dir, target_n, args, view_type=row.get("view_type", ""))
    elif method == "region_flow_blend":
        written, stats = generate_region_flow_blend(cv2, np, frames, out_dir, target_n, args, view_type=row.get("view_type", ""))
    elif method == "region_norm_medium":
        history_gap = 1
        if len(frame_paths) >= 2:
            history_gap = max(1, int(frame_paths[-1].stem) - int(frame_paths[-2].stem))
        written, stats = generate_region_norm_medium(
            cv2,
            np,
            frames,
            out_dir,
            target_n,
            args,
            view_type=row.get("view_type", ""),
            history_gap=history_gap,
        )
    elif method == "actor_layer_small_filled":
        history_gap = 1
        if len(frame_paths) >= 2:
            history_gap = max(1, int(frame_paths[-1].stem) - int(frame_paths[-2].stem))
        written, stats = generate_actor_layer_small_filled(
            cv2,
            np,
            frames,
            out_dir,
            target_n,
            args,
            history_gap=history_gap,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    vr = verify_sample_output(out_dir, target_n)
    base["frames_written"] = written
    base["status"] = "ok" if vr.ok and written == target_n else "bad"
    note_items = []
    if base["notes"]:
        note_items.append(base["notes"])
    if vr.notes:
        note_items.append(vr.notes)
    if stats:
        note = "motion_mean={:.4f},motion_p90={:.4f},motion_max={:.4f}".format(
            stats["mean_motion"],
            stats["p90_motion"],
            stats["max_motion"],
        )
        if "gate" in stats:
            note += ",gate={:.4f}".format(stats["gate"])
        if "mask_mean" in stats:
            note += ",mask_mean={:.4f},mask_p90={:.4f}".format(stats["mask_mean"], stats["mask_p90"])
        if "actor_guard_mean" in stats:
            note += ",actor_guard_mean={:.4f},actor_guard_p90={:.4f},actor_guard_strength={:.2f}".format(
                stats["actor_guard_mean"],
                stats["actor_guard_p90"],
                stats["actor_guard_strength"],
            )
        note_items.append(note)
    if args.method == "view_hybrid":
        note_items.append(f"resolved_method={method}")
    base["notes"] = "; ".join(note_items)

    if args.make_montage and base["status"] == "ok":
        montage_dir = Path(args.out) / "_montages"
        make_montage(cv2, np, row, frames, out_dir, montage_dir / f"{output_name}.png")

    return base


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_view: dict[str, int] = {}
    total_frames = 0
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        by_view[row["view_type"]] = by_view.get(row["view_type"], 0) + 1
        total_frames += parse_int(str(row.get("frames_written", "")), 0)
    return {
        "rows": len(rows),
        "by_status": by_status,
        "by_view": by_view,
        "frames_written": total_frames,
    }


def run(args) -> None:
    index_dir = Path(args.index_dir).expanduser()
    out_dir = Path(args.out).expanduser()
    rows = load_test_rows(index_dir)
    if args.view_type:
        rows = [row for row in rows if row.get("view_type", "") == args.view_type]
    if args.phase != "":
        rows = [row for row in rows if row.get("phase", "") == str(args.phase)]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    cv2 = None
    np = None
    if not args.dry_run and not args.verify_only:
        cv2, np = import_cv2_np()

    used_names: set[str] = set()
    output_names = [resolve_output_name(row, used_names) for row in rows]

    section("TRACK 5 BASELINE END TO END")
    kv("index_dir", index_dir)
    kv("test_csv", index_dir / "test_index.csv")
    kv("out_dir", out_dir)
    kv("rows", len(rows))
    kv("view_type", args.view_type or "(all)")
    kv("method", args.method)
    kv("dry_run", args.dry_run)
    kv("verify_only", args.verify_only)
    kv("clean outputs", args.clean and not args.verify_only and not args.dry_run)
    kv("make_montage", args.make_montage)
    if args.method == "flow_decay":
        kv("flow_width", args.flow_width)
        kv("flow_decay", args.flow_decay)
        kv("max_flow_scale", args.max_flow_scale)
        kv("last_anchor_decay", args.last_anchor_decay)
        kv("min_last_anchor", args.min_last_anchor)

    manifest_rows = []
    failures = []
    for row, output_name in zip(rows, output_names):
        if not args.dry_run and not args.verify_only and (cv2 is None or np is None):
            raise SystemExit("OpenCV/numpy import failed before generation.")
        item = process_row(cv2, np, row, output_name, args)
        manifest_rows.append(item)
        if item["status"] not in ("ok", "dry_run"):
            failures.append(item)

    manifest_path = out_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    summary = summarize(manifest_rows)
    summary_path = out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    section("RESULT")
    kv("manifest", manifest_path)
    kv("summary", summary_path)
    kv("rows", summary["rows"])
    kv("frames_written", summary["frames_written"])
    kv("by_status", summary["by_status"])
    kv("by_view", summary["by_view"])
    kv("failures", len(failures))
    if args.make_montage:
        kv("montage_dir", out_dir / "_montages")

    if failures:
        section("FIRST FAILURES")
        for item in failures[:20]:
            print(f"  - {item['source_folder']} | {item['status']} | {item['notes']}")
        raise SystemExit(2)
