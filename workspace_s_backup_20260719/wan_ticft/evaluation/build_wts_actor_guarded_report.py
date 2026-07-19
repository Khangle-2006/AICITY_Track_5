from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("status") == "ok"}


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def passes_guard(row: dict[str, str], args: argparse.Namespace) -> bool:
    return (
        parse_float(row, "actor_mask_sum_frac") <= args.max_sum_frac
        and parse_float(row, "actor_mask_mean") <= args.max_mean
        and parse_float(row, "actor_mask_max_frac") <= args.max_max_frac
        and parse_float(row, "actor_component_count") <= args.max_components
        and parse_float(row, "actor_raw_sum_frac") <= args.max_raw_sum_frac
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Route WTS overhead actor-layer only when mask stats are safe.")
    parser.add_argument("--current_report", type=Path, required=True)
    parser.add_argument("--actor_report", type=Path, required=True)
    parser.add_argument("--out_report", type=Path, required=True)
    parser.add_argument("--route_label", required=True)
    parser.add_argument("--max_sum_frac", type=float, required=True)
    parser.add_argument("--max_mean", type=float, required=True)
    parser.add_argument("--max_max_frac", type=float, default=0.020)
    parser.add_argument("--max_components", type=float, default=80)
    parser.add_argument("--max_raw_sum_frac", type=float, default=0.35)
    args = parser.parse_args()

    current = read_rows(args.current_report)
    actor = read_rows(args.actor_report)
    rows = []
    selected = 0
    overhead = 0
    for sample_id, base in current.items():
        row = dict(base)
        use_actor = False
        if base.get("view_type") == "wts_overhead_or_fixed_camera":
            overhead += 1
            actor_row = actor.get(sample_id)
            if actor_row is not None and passes_guard(actor_row, args):
                row = dict(actor_row)
                use_actor = True
                selected += 1
        row["route"] = args.route_label
        row["guard_selected_actor"] = "1" if use_actor else "0"
        row["guard_max_sum_frac"] = f"{args.max_sum_frac:.6f}"
        row["guard_max_mean"] = f"{args.max_mean:.6f}"
        rows.append(row)

    if not rows:
        raise RuntimeError("no rows selected")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with args.out_report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)}")
    print(f"overhead_rows={overhead}")
    print(f"selected_actor={selected}")
    print(f"output={args.out_report}")


if __name__ == "__main__":
    main()
