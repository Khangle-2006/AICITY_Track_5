from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .console import kv, section
from .evaluate_val import choose_rows, filter_rows, process_row
from .frame_io import import_cv2_np
from .index_io import read_csv


FLOW_CONFIGS = [
    {"name": "flow_current", "flow_decay": 0.92, "max_flow_scale": 12.0, "last_anchor_decay": 0.970, "min_last_anchor": 0.15},
    {"name": "flow_balanced", "flow_decay": 0.90, "max_flow_scale": 8.0, "last_anchor_decay": 0.975, "min_last_anchor": 0.25},
    {"name": "flow_conservative", "flow_decay": 0.88, "max_flow_scale": 6.0, "last_anchor_decay": 0.980, "min_last_anchor": 0.35},
    {"name": "flow_strong", "flow_decay": 0.95, "max_flow_scale": 12.0, "last_anchor_decay": 0.970, "min_last_anchor": 0.15},
]

LIGHT_CONFIGS = [
    {"name": "light_current", "light_flow_decay": 0.80, "light_max_flow_scale": 4.0, "light_last_anchor_decay": 0.985, "light_min_last_anchor": 0.45},
    {"name": "light_lighter", "light_flow_decay": 0.75, "light_max_flow_scale": 3.0, "light_last_anchor_decay": 0.990, "light_min_last_anchor": 0.55},
    {"name": "light_repeatish", "light_flow_decay": 0.70, "light_max_flow_scale": 2.0, "light_last_anchor_decay": 0.995, "light_min_last_anchor": 0.65},
    {"name": "light_medium", "light_flow_decay": 0.85, "light_max_flow_scale": 5.0, "light_last_anchor_decay": 0.985, "light_min_last_anchor": 0.40},
]


def make_args(base, method: str, flow_cfg: dict[str, Any], light_cfg: dict[str, Any] | None = None):
    values = vars(base).copy()
    values.update(flow_cfg)
    if light_cfg:
        values.update(light_cfg)
    else:
        values.update(LIGHT_CONFIGS[0])
    values["methods"] = [method]
    return SimpleNamespace(**values)


def mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    nums = [float(r[metric]) for r in rows if r.get(metric, "") != ""]
    return sum(nums) / len(nums) if nums else 0.0


def evaluate_config(cv2, np, selected: list[dict[str, str]], args, config_name: str) -> dict[str, Any]:
    metric_rows = []
    for row in selected:
        metric_rows.extend(process_row(cv2, np, row, args))
    valid = [r for r in metric_rows if r.get("psnr", "") != ""]
    psnr = mean_metric(valid, "psnr")
    ssim = mean_metric(valid, "ssim")
    absdiff = mean_metric(valid, "absdiff")
    return {
        "config": config_name,
        "method": args.methods[0],
        "n": len(valid),
        "psnr": psnr,
        "ssim": ssim,
        "absdiff": absdiff,
        "flow_decay": args.flow_decay,
        "max_flow_scale": args.max_flow_scale,
        "last_anchor_decay": args.last_anchor_decay,
        "min_last_anchor": args.min_last_anchor,
        "light_flow_decay": args.light_flow_decay,
        "light_max_flow_scale": args.light_max_flow_scale,
        "light_last_anchor_decay": args.light_last_anchor_decay,
        "light_min_last_anchor": args.light_min_last_anchor,
    }


def add_scores(rows: list[dict[str, Any]], ssim_weight: float, abs_weight: float) -> None:
    for row in rows:
        row["score"] = float(row["psnr"]) + ssim_weight * float(row["ssim"]) - abs_weight * float(row["absdiff"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "config",
        "method",
        "n",
        "score",
        "psnr",
        "ssim",
        "absdiff",
        "flow_decay",
        "max_flow_scale",
        "last_anchor_decay",
        "min_last_anchor",
        "light_flow_decay",
        "light_max_flow_scale",
        "light_last_anchor_decay",
        "light_min_last_anchor",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def print_top(rows: list[dict[str, Any]], key: str, reverse: bool = True, n: int = 10) -> None:
    print()
    print(f"Top by {key}")
    print("-" * 132)
    print(f"{'Config':<34} {'Method':<12} {'Score':>9} {'PSNR':>9} {'SSIM':>9} {'AbsDiff':>9} {'Flow':>20} {'Light':>20}")
    print("-" * 132)
    for row in sorted(rows, key=lambda x: float(x[key]), reverse=reverse)[:n]:
        flow = f"{row['flow_decay']}/{row['max_flow_scale']}/{row['min_last_anchor']}"
        light = f"{row['light_flow_decay']}/{row['light_max_flow_scale']}/{row['light_min_last_anchor']}"
        print(
            f"{row['config']:<34} {row['method']:<12} {row['score']:>9.4f} "
            f"{row['psnr']:>9.4f} {row['ssim']:>9.4f} {row['absdiff']:>9.4f} "
            f"{flow:>20} {light:>20}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--out_csv", default="/workspace/experiments/baseline_tune_classical.csv")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--dataset_part", default="")
    parser.add_argument("--view_type", default="")
    parser.add_argument("--phase", default="")
    parser.add_argument("--per_group", type=int, default=5)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--target_gap_frames", type=int, default=4)
    parser.add_argument("--eval_horizon", type=int, default=4)
    parser.add_argument("--flow_width", type=int, default=426)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--ssim_weight", type=float, default=10.0)
    parser.add_argument("--abs_weight", type=float, default=0.10)
    parser.add_argument("--include_flow_decay", action="store_true", default=True)
    parser.add_argument("--include_view_hybrid", action="store_true", default=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    index_dir = Path(args.index_dir).expanduser()
    csv_path = index_dir / f"{args.split}_index.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing index CSV: {csv_path}")
    rows = filter_rows(read_csv(csv_path), args)
    selected = choose_rows(rows, args.per_group, args.seed)

    section("TUNE CLASSICAL BASELINE")
    kv("index_dir", index_dir)
    kv("csv", csv_path)
    kv("out_csv", args.out_csv)
    kv("rows after filter", len(rows))
    kv("selected rows", len(selected))
    kv("target_gap_frames", args.target_gap_frames)
    kv("eval_horizon", args.eval_horizon)
    kv("score", f"psnr + {args.ssim_weight}*ssim - {args.abs_weight}*absdiff")

    results: list[dict[str, Any]] = []
    if args.include_flow_decay:
        for flow_cfg in FLOW_CONFIGS:
            cfg_args = make_args(args, "flow_decay", flow_cfg)
            results.append(evaluate_config(cv2, np, selected, cfg_args, flow_cfg["name"]))

    if args.include_view_hybrid:
        for flow_cfg in FLOW_CONFIGS:
            for light_cfg in LIGHT_CONFIGS:
                cfg_args = make_args(args, "view_hybrid", flow_cfg, light_cfg)
                name = f"hybrid_{flow_cfg['name']}_{light_cfg['name']}"
                results.append(evaluate_config(cv2, np, selected, cfg_args, name))

    add_scores(results, args.ssim_weight, args.abs_weight)
    out_csv = Path(args.out_csv).expanduser()
    write_csv(out_csv, results)

    section("RESULT")
    kv("configs", len(results))
    kv("out_csv", out_csv)
    print_top(results, "score", reverse=True)
    print_top(results, "psnr", reverse=True)
    print_top(results, "absdiff", reverse=False)


if __name__ == "__main__":
    main()
