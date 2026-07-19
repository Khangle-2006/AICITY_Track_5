from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

from .console import kv, section
from .frame_io import import_cv2_np
from .generators import compute_flow, flow_decay_scales, make_region_motion_mask, motion_gate, warp_rgb
from .index_io import read_csv
from .metrics import mean_absdiff, psnr, ssim_gray
from .video_io import extract_val_history_and_target


CSV_FIELDS = [
    "sample_id",
    "dataset_part",
    "view_type",
    "phase",
    "method",
    "target_gap_frames",
    "psnr",
    "ssim",
    "absdiff",
    "notes",
]


def filter_rows(rows: list[dict[str, str]], args) -> list[dict[str, str]]:
    out = rows
    for key in ("dataset_part", "view_type", "phase"):
        val = getattr(args, key)
        if val:
            out = [r for r in out if str(r.get(key, "")) == str(val)]
    return out


def group_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("dataset_part", ""), row.get("view_type", ""), row.get("phase", ""))


def choose_rows(rows: list[dict[str, str]], per_group: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    selected = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r.get("sample_id", ""))
        if per_group <= 0 or len(group) <= per_group:
            selected.extend(group)
        else:
            selected.extend(rng.sample(group, per_group))
    return selected


def predict_repeat(frames, target_n: int):
    return [frames[-1] for _ in range(target_n)]


def predict_flow_decay(cv2, np, frames, target_n: int, args):
    if len(frames) < 2:
        return predict_repeat(frames, target_n)
    last = frames[-1]
    prev = frames[-2]
    flow = compute_flow(cv2, np, prev, last, args.flow_width)
    scales = flow_decay_scales(target_n, args.flow_decay, args.max_flow_scale)
    preds = []
    for i, scale in enumerate(scales):
        pred = warp_rgb(cv2, np, last, flow, scale)
        anchor = max(args.min_last_anchor, args.last_anchor_decay ** (i + 1))
        if anchor > 0:
            pred_f = pred.astype(np.float32)
            last_f = last.astype(np.float32)
            pred = np.clip(anchor * last_f + (1.0 - anchor) * pred_f, 0, 255).astype(np.uint8)
        preds.append(pred)
    return preds


def predict_flow_light(cv2, np, frames, target_n: int, args):
    if len(frames) < 2:
        return predict_repeat(frames, target_n)
    last = frames[-1]
    prev = frames[-2]
    flow = compute_flow(cv2, np, prev, last, args.flow_width)
    scales = flow_decay_scales(target_n, args.light_flow_decay, args.light_max_flow_scale)
    preds = []
    for i, scale in enumerate(scales):
        pred = warp_rgb(cv2, np, last, flow, scale)
        anchor = max(args.light_min_last_anchor, args.light_last_anchor_decay ** (i + 1))
        if anchor > 0:
            pred_f = pred.astype(np.float32)
            last_f = last.astype(np.float32)
            pred = np.clip(anchor * last_f + (1.0 - anchor) * pred_f, 0, 255).astype(np.uint8)
        preds.append(pred)
    return preds


def predict_motion_adaptive(cv2, np, frames, target_n: int, args, view_type: str):
    if len(frames) < 2:
        return predict_repeat(frames, target_n)
    last = frames[-1]
    prev = frames[-2]
    flow = compute_flow(cv2, np, prev, last, args.flow_width)
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    gate = motion_gate(np, float(np.percentile(mag, 90)), view_type, args)
    scales = flow_decay_scales(target_n, args.flow_decay, args.max_flow_scale)
    preds = []
    for scale in scales:
        flow_pred = warp_rgb(cv2, np, last, flow, scale)
        pred_f = gate * flow_pred.astype(np.float32) + (1.0 - gate) * last.astype(np.float32)
        preds.append(np.clip(pred_f, 0, 255).astype(np.uint8))
    return preds


def predict_region_flow_blend(cv2, np, frames, target_n: int, args, view_type: str):
    if len(frames) < 2:
        return predict_repeat(frames, target_n)
    last = frames[-1]
    prev = frames[-2]
    flow = compute_flow(cv2, np, prev, last, args.flow_width)
    base_mask, _ = make_region_motion_mask(cv2, np, flow, view_type, args)
    if view_type == "wts_overhead_or_fixed_camera":
        max_scale = args.region_overhead_max_flow_scale
    else:
        max_scale = args.region_max_flow_scale
    scales = flow_decay_scales(target_n, args.region_flow_decay, max_scale)
    preds = []
    last_f = last.astype(np.float32)
    for i, scale in enumerate(scales):
        flow_pred = warp_rgb(cv2, np, last, flow, scale).astype(np.float32)
        horizon_gain = min(1.0, args.region_horizon_gain_base + args.region_horizon_gain_step * i)
        mask = np.clip(base_mask * horizon_gain, 0.0, 1.0)[:, :, None]
        preds.append(np.clip((1.0 - mask) * last_f + mask * flow_pred, 0, 255).astype(np.uint8))
    return preds


def evaluate_prediction(cv2, np, pred, target) -> dict[str, float]:
    if pred.shape[:2] != target.shape[:2]:
        target = cv2.resize(target, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)
    return {
        "psnr": psnr(cv2, np, pred, target),
        "ssim": ssim_gray(cv2, np, pred, target),
        "absdiff": mean_absdiff(np, pred, target),
    }


def process_row(cv2, np, row: dict[str, str], args) -> list[dict[str, Any]]:
    frames, target, meta = extract_val_history_and_target(cv2, row, args.num_history, args.target_gap_frames)
    if meta and "error" in meta:
        return [
            {
                "sample_id": row.get("sample_id", ""),
                "dataset_part": row.get("dataset_part", ""),
                "view_type": row.get("view_type", ""),
                "phase": row.get("phase", ""),
                "method": method,
                "target_gap_frames": args.target_gap_frames,
                "notes": meta["error"],
            }
            for method in args.methods
        ]
    if not frames or target is None:
        return [
            {
                "sample_id": row.get("sample_id", ""),
                "dataset_part": row.get("dataset_part", ""),
                "view_type": row.get("view_type", ""),
                "phase": row.get("phase", ""),
                "method": method,
                "target_gap_frames": args.target_gap_frames,
                "notes": "missing frames or target",
            }
            for method in args.methods
        ]

    out = []
    horizon = max(1, args.eval_horizon)
    for method in args.methods:
        if method == "repeat":
            preds = predict_repeat(frames, horizon)
        elif method == "flow_decay":
            preds = predict_flow_decay(cv2, np, frames, horizon, args)
        elif method == "flow_light":
            preds = predict_flow_light(cv2, np, frames, horizon, args)
        elif method == "view_hybrid":
            if row.get("view_type", "") == "wts_overhead_or_fixed_camera":
                preds = predict_flow_light(cv2, np, frames, horizon, args)
            else:
                preds = predict_flow_decay(cv2, np, frames, horizon, args)
        elif method == "motion_adaptive":
            preds = predict_motion_adaptive(cv2, np, frames, horizon, args, row.get("view_type", ""))
        elif method == "region_flow_blend":
            preds = predict_region_flow_blend(cv2, np, frames, horizon, args, row.get("view_type", ""))
        else:
            raise ValueError(f"Unknown method: {method}")

        pred = preds[-1]
        m = evaluate_prediction(cv2, np, pred, target)
        out.append(
            {
                "sample_id": row.get("sample_id", ""),
                "dataset_part": row.get("dataset_part", ""),
                "view_type": row.get("view_type", ""),
                "phase": row.get("phase", ""),
                "method": method,
                "target_gap_frames": args.target_gap_frames,
                "psnr": m["psnr"],
                "ssim": m["ssim"],
                "absdiff": m["absdiff"],
                "notes": f"eval_horizon={horizon}",
            }
        )
    return out


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def aggregate(rows: list[dict[str, Any]], group_keys: list[str]) -> dict[tuple[str, ...], dict[str, float]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("psnr", "") == "":
            continue
        key = tuple(str(row.get(k, "")) for k in group_keys)
        groups.setdefault(key, []).append(row)

    out = {}
    for key, items in groups.items():
        vals = {"n": float(len(items))}
        for metric in ("psnr", "ssim", "absdiff"):
            nums = [float(x[metric]) for x in items if x.get(metric, "") != ""]
            vals[metric] = sum(nums) / len(nums) if nums else 0.0
        out[key] = vals
    return out


def print_table(title: str, agg: dict[tuple[str, ...], dict[str, float]]) -> None:
    print()
    print(title)
    print("-" * 110)
    print(f"{'Group':<62} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 110)
    for key, vals in sorted(agg.items()):
        group = "/".join(key)
        print(f"{group[:62]:<62} {int(vals['n']):>5} {vals['psnr']:>10.4f} {vals['ssim']:>10.4f} {vals['absdiff']:>10.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--out_csv", default="/workspace/experiments/baseline_val_metrics.csv")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--dataset_part", default="")
    parser.add_argument("--view_type", default="")
    parser.add_argument("--phase", default="")
    parser.add_argument("--per_group", type=int, default=5, help="Rows per dataset/view/phase group. 0 means all.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["repeat", "flow_decay"],
        choices=["repeat", "flow_decay", "flow_light", "view_hybrid", "motion_adaptive", "region_flow_blend"],
    )
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--target_gap_frames", type=int, default=4)
    parser.add_argument("--eval_horizon", type=int, default=4, help="Predict this many steps and score the last prediction.")
    parser.add_argument("--flow_width", type=int, default=426)
    parser.add_argument("--flow_decay", type=float, default=0.92)
    parser.add_argument("--max_flow_scale", type=float, default=12.0)
    parser.add_argument("--last_anchor_decay", type=float, default=0.97)
    parser.add_argument("--min_last_anchor", type=float, default=0.15)
    parser.add_argument("--light_flow_decay", type=float, default=0.80)
    parser.add_argument("--light_max_flow_scale", type=float, default=4.0)
    parser.add_argument("--light_last_anchor_decay", type=float, default=0.985)
    parser.add_argument("--light_min_last_anchor", type=float, default=0.45)
    parser.add_argument("--motion_threshold", type=float, default=2.0)
    parser.add_argument("--motion_temperature", type=float, default=1.0)
    parser.add_argument("--motion_min_gate", type=float, default=0.05)
    parser.add_argument("--motion_max_gate", type=float, default=0.95)
    parser.add_argument("--motion_overhead_gate_scale", type=float, default=0.65)
    parser.add_argument("--region_motion_tau", type=float, default=1.5)
    parser.add_argument("--region_motion_temp", type=float, default=0.75)
    parser.add_argument("--region_max_alpha", type=float, default=0.85)
    parser.add_argument("--region_overhead_max_alpha", type=float, default=0.45)
    parser.add_argument("--region_mask_blur", type=float, default=3.0)
    parser.add_argument("--region_flow_decay", type=float, default=0.92)
    parser.add_argument("--region_max_flow_scale", type=float, default=10.0)
    parser.add_argument("--region_overhead_max_flow_scale", type=float, default=4.0)
    parser.add_argument("--region_horizon_gain_base", type=float, default=0.85)
    parser.add_argument("--region_horizon_gain_step", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=11)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    index_dir = Path(args.index_dir).expanduser()
    csv_path = index_dir / f"{args.split}_index.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing index CSV: {csv_path}")
    rows = read_csv(csv_path)
    rows = filter_rows(rows, args)
    selected = choose_rows(rows, args.per_group, args.seed)

    section("TRACK 5 VALIDATION BASELINE EVAL")
    kv("index_dir", index_dir)
    kv("csv", csv_path)
    kv("out_csv", args.out_csv)
    kv("rows after filter", len(rows))
    kv("selected rows", len(selected))
    kv("methods", args.methods)
    kv("target_gap_frames", args.target_gap_frames)
    kv("eval_horizon", args.eval_horizon)
    kv("num_history", args.num_history)

    metric_rows: list[dict[str, Any]] = []
    for row in selected:
        metric_rows.extend(process_row(cv2, np, row, args))

    out_csv = Path(args.out_csv).expanduser()
    write_metrics(out_csv, metric_rows)
    valid_rows = [r for r in metric_rows if r.get("psnr", "") != ""]
    failures = [r for r in metric_rows if r.get("psnr", "") == ""]

    section("RESULT")
    kv("metrics", out_csv)
    kv("metric rows", len(metric_rows))
    kv("valid rows", len(valid_rows))
    kv("failures", len(failures))

    section("AGGREGATE")
    print_table("By method", aggregate(valid_rows, ["method"]))
    print_table("By method/view", aggregate(valid_rows, ["method", "view_type"]))
    print_table("By method/dataset/view/phase", aggregate(valid_rows, ["method", "dataset_part", "view_type", "phase"]))

    if failures:
        section("FIRST FAILURES")
        for row in failures[:20]:
            print(f"  - {row.get('sample_id')} | {row.get('method')} | {row.get('notes')}")


if __name__ == "__main__":
    main()
