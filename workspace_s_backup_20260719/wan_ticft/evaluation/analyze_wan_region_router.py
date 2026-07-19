from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan_eval_csv", type=Path, required=True)
    parser.add_argument("--motion_csv", type=Path, required=True)
    parser.add_argument("--wan_label", default="wan")
    parser.add_argument("--region_label", default="region_norm_medium")
    parser.add_argument("--out_prefix", type=Path, required=True)
    parser.add_argument("--min_group_delta", type=float, default=0.1)
    parser.add_argument("--min_group_win_rate", type=float, default=0.6)
    args = parser.parse_args()

    wan = {
        row["sample_id"]: row
        for row in read_csv(args.wan_eval_csv)
        if row.get("candidate") == args.wan_label and row.get("status") == "ok"
    }
    region = {
        row["sample_id"]: row
        for row in read_csv(args.motion_csv)
        if row.get("candidate") == args.region_label and row.get("status") == "ok"
    }

    samples: list[dict[str, object]] = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for sample_id in sorted(set(wan) & set(region)):
        w = wan[sample_id]
        r = region[sample_id]
        delta_psnr = float(r["psnr"]) - float(w["psnr"])
        delta_ssim = float(r["ssim"]) - float(w["ssim"])
        row: dict[str, object] = {
            "sample_id": sample_id,
            "view_type": w.get("view_type", ""),
            "phase": w.get("phase", ""),
            "wan_psnr": float(w["psnr"]),
            "region_psnr": float(r["psnr"]),
            "delta_psnr": delta_psnr,
            "wan_ssim": float(w["ssim"]),
            "region_ssim": float(r["ssim"]),
            "delta_ssim": delta_ssim,
            "winner": "region" if delta_psnr > 0 else "wan",
        }
        samples.append(row)
        groups[(str(row["view_type"]), str(row["phase"]))].append(row)

    group_rows: list[dict[str, object]] = []
    recommended: list[str] = []
    for (view_type, phase), items in sorted(groups.items()):
        delta_psnr = mean(float(item["delta_psnr"]) for item in items)
        delta_ssim = mean(float(item["delta_ssim"]) for item in items)
        wins = sum(item["winner"] == "region" for item in items)
        win_rate = wins / len(items)
        use_region = delta_psnr >= args.min_group_delta and win_rate >= args.min_group_win_rate
        if use_region:
            recommended.append(f"{view_type}:{phase}")
        group_rows.append(
            {
                "view_type": view_type,
                "phase": phase,
                "n": len(items),
                "region_wins": wins,
                "region_win_rate": win_rate,
                "mean_delta_psnr": delta_psnr,
                "mean_delta_ssim": delta_ssim,
                "recommended": use_region,
            }
        )

    sample_path = args.out_prefix.with_name(args.out_prefix.name + "_samples.csv")
    group_path = args.out_prefix.with_name(args.out_prefix.name + "_groups.csv")
    json_path = args.out_prefix.with_name(args.out_prefix.name + "_recommendation.json")
    write_csv(
        sample_path,
        samples,
        [
            "sample_id",
            "view_type",
            "phase",
            "wan_psnr",
            "region_psnr",
            "delta_psnr",
            "wan_ssim",
            "region_ssim",
            "delta_ssim",
            "winner",
        ],
    )
    write_csv(
        group_path,
        group_rows,
        [
            "view_type",
            "phase",
            "n",
            "region_wins",
            "region_win_rate",
            "mean_delta_psnr",
            "mean_delta_ssim",
            "recommended",
        ],
    )
    json_path.write_text(
        json.dumps(
            {
                "matched_samples": len(samples),
                "recommended_region_view_phases": recommended,
                "min_group_delta": args.min_group_delta,
                "min_group_win_rate": args.min_group_win_rate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"matched samples: {len(samples)}")
    print("recommended region routes:")
    for value in recommended:
        print(f"  {value}")
    print(f"samples: {sample_path}")
    print(f"groups: {group_path}")
    print(f"recommendation: {json_path}")


if __name__ == "__main__":
    main()
