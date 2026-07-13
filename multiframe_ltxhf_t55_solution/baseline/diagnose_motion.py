from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .console import kv, section
from .frame_io import import_cv2_np, numeric_pngs, read_rgb, resize_if_needed
from .index_io import load_test_rows
from .naming import parse_int, resolve_output_name


def mean_abs(a, b, np) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def gray_edges(cv2, rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_32F)


def edge_change(cv2, a, b, np) -> float:
    ea = gray_edges(cv2, a)
    eb = gray_edges(cv2, b)
    return float(np.mean(np.abs(ea - eb)))


def sample_pair_indices(n: int, stride: int) -> list[tuple[int, int]]:
    if n < 2:
        return []
    stride = max(1, stride)
    return [(i, i + stride) for i in range(0, n - stride)]


def load_frames(cv2, frame_dir: Path, resize_width: int, resize_height: int):
    frames = []
    paths = numeric_pngs(frame_dir)
    for path in paths:
        img = read_rgb(cv2, path)
        if img is None:
            continue
        frames.append(resize_if_needed(cv2, img, resize_width, resize_height))
    return frames, paths


def diagnose_sample(cv2, np, item: dict[str, Any], gen_dir: Path, history_stride: int, pred_stride: int, resize_width: int, resize_height: int) -> dict[str, Any]:
    out_dir = gen_dir / item["output_name"]
    pred_frames, pred_paths = load_frames(cv2, out_dir, resize_width, resize_height)

    hist_dir = Path(item.get("input_dir", ""))
    hist_frames, _ = load_frames(cv2, hist_dir, resize_width, resize_height)

    pred_diffs = []
    pred_edge_diffs = []
    for i, j in sample_pair_indices(len(pred_frames), pred_stride):
        pred_diffs.append(mean_abs(pred_frames[i], pred_frames[j], np))
        pred_edge_diffs.append(edge_change(cv2, pred_frames[i], pred_frames[j], np))

    hist_diffs = []
    for i, j in sample_pair_indices(len(hist_frames), history_stride):
        hist_diffs.append(mean_abs(hist_frames[i], hist_frames[j], np))

    first_last_abs = mean_abs(pred_frames[0], pred_frames[-1], np) if len(pred_frames) >= 2 else 0.0
    hist_mean = float(np.mean(hist_diffs)) if hist_diffs else 0.0
    pred_mean = float(np.mean(pred_diffs)) if pred_diffs else 0.0
    pred_p90 = float(np.percentile(pred_diffs, 90)) if pred_diffs else 0.0
    edge_mean = float(np.mean(pred_edge_diffs)) if pred_edge_diffs else 0.0
    motion_ratio = pred_mean / (hist_mean + 1e-6) if hist_mean > 0 else 0.0

    freeze_flag = int(pred_mean < 1.0 or first_last_abs < 2.0)
    weak_motion_flag = int(pred_mean < 2.0 or motion_ratio < 0.35)

    return {
        "output_name": item["output_name"],
        "sample_id": item.get("sample_id", ""),
        "source_folder": item.get("source_folder", ""),
        "dataset_part": item.get("dataset_part", ""),
        "view_type": item.get("view_type", ""),
        "target_N": item.get("target_N", ""),
        "pred_frames": len(pred_paths),
        "hist_frames": len(hist_frames),
        "pred_temporal_abs_mean": pred_mean,
        "pred_temporal_abs_p90": pred_p90,
        "pred_first_last_abs": first_last_abs,
        "pred_edge_change_mean": edge_mean,
        "history_temporal_abs_mean": hist_mean,
        "pred_vs_history_motion_ratio": motion_ratio,
        "freeze_flag": freeze_flag,
        "weak_motion_flag": weak_motion_flag,
        "status": "ok" if len(pred_frames) >= 2 else "bad",
        "notes": "" if len(pred_frames) >= 2 else "not enough readable frames",
    }


def build_expected(index_dir: Path) -> list[dict[str, Any]]:
    rows = load_test_rows(index_dir)
    used: set[str] = set()
    out = []
    for row in rows:
        output_name = resolve_output_name(row, used)
        out.append({**row, "output_name": output_name, "target_N": parse_int(row.get("target_N", ""), 0)})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "output_name",
        "sample_id",
        "source_folder",
        "dataset_part",
        "view_type",
        "target_N",
        "pred_frames",
        "hist_frames",
        "pred_temporal_abs_mean",
        "pred_temporal_abs_p90",
        "pred_first_last_abs",
        "pred_edge_change_mean",
        "history_temporal_abs_mean",
        "pred_vs_history_motion_ratio",
        "freeze_flag",
        "weak_motion_flag",
        "status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def aggregate(rows: list[dict[str, Any]], key: str, np) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)

    out = []
    for name, items in sorted(groups.items()):
        ok = [r for r in items if r["status"] == "ok"]
        if not ok:
            continue
        out.append(
            {
                "group": name,
                "n": len(ok),
                "pred_temporal_abs_mean": float(np.mean([r["pred_temporal_abs_mean"] for r in ok])),
                "pred_first_last_abs": float(np.mean([r["pred_first_last_abs"] for r in ok])),
                "history_temporal_abs_mean": float(np.mean([r["history_temporal_abs_mean"] for r in ok])),
                "motion_ratio": float(np.mean([r["pred_vs_history_motion_ratio"] for r in ok])),
                "freeze_rate": float(np.mean([r["freeze_flag"] for r in ok])),
                "weak_motion_rate": float(np.mean([r["weak_motion_flag"] for r in ok])),
            }
        )
    return out


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print()
    print(title)
    print("-" * 112)
    print(f"{'Group':<40} {'N':>5} {'PredDiff':>10} {'FirstLast':>10} {'HistDiff':>10} {'Ratio':>8} {'Freeze':>8} {'Weak':>8}")
    print("-" * 112)
    for row in rows:
        print(
            f"{row['group']:<40} {row['n']:>5} "
            f"{row['pred_temporal_abs_mean']:>10.3f} "
            f"{row['pred_first_last_abs']:>10.3f} "
            f"{row['history_temporal_abs_mean']:>10.3f} "
            f"{row['motion_ratio']:>8.3f} "
            f"{row['freeze_rate']:>8.2%} "
            f"{row['weak_motion_rate']:>8.2%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose temporal motion/freeze in generated predictions.")
    parser.add_argument("--index_dir", default="/workspace/experiments/index", type=Path)
    parser.add_argument("--gen_dir", required=True, type=Path)
    parser.add_argument("--out_csv", required=True, type=Path)
    parser.add_argument("--resize_width", type=int, default=320)
    parser.add_argument("--resize_height", type=int, default=180)
    parser.add_argument("--history_stride", type=int, default=1)
    parser.add_argument("--pred_stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    expected = build_expected(args.index_dir)
    if args.limit > 0:
        expected = expected[: args.limit]

    section("DIAGNOSE PREDICTION MOTION")
    kv("index_dir", args.index_dir)
    kv("gen_dir", args.gen_dir)
    kv("out_csv", args.out_csv)
    kv("samples", len(expected))
    kv("resize", f"{args.resize_width}x{args.resize_height}")

    rows = [
        diagnose_sample(
            cv2,
            np,
            item,
            args.gen_dir,
            args.history_stride,
            args.pred_stride,
            args.resize_width,
            args.resize_height,
        )
        for item in expected
    ]
    write_csv(args.out_csv, rows)

    section("AGGREGATE")
    ok = [r for r in rows if r["status"] == "ok"]
    kv("rows", len(rows))
    kv("ok", len(ok))
    kv("bad", len(rows) - len(ok))
    if ok:
        print_table("By view_type", aggregate(ok, "view_type", np))
        print_table("By dataset_part", aggregate(ok, "dataset_part", np))

    section("MOST FROZEN")
    frozen = sorted(ok, key=lambda r: (r["pred_first_last_abs"], r["pred_temporal_abs_mean"]))[:15]
    print(f"{'Output':<52} {'View':<30} {'PredDiff':>10} {'FirstLast':>10} {'Ratio':>8}")
    print("-" * 116)
    for row in frozen:
        print(
            f"{row['output_name']:<52} {row['view_type']:<30} "
            f"{row['pred_temporal_abs_mean']:>10.3f} "
            f"{row['pred_first_last_abs']:>10.3f} "
            f"{row['pred_vs_history_motion_ratio']:>8.3f}"
        )

    section("OUTPUT")
    kv("motion csv", args.out_csv)


if __name__ == "__main__":
    main()
