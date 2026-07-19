from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from baseline.frame_io import numeric_pngs
from baseline.verify import verify_sample_output


MODES = {
    # Preserve pixel score first: only use a125 when the depth renderer is stable and
    # actor occupancy is moderate. This is the public-score rescue candidate.
    "strict": {
        "min_coverage": 0.72,
        "min_pose_quality": 0.53,
        "min_confidence": 0.18,
        "min_actor_fraction": 0.04,
        "max_actor_fraction": 0.30,
    },
    # Slightly more a125 exposure; useful if strict is too close to depth_a75.
    "balanced": {
        "min_coverage": 0.68,
        "min_pose_quality": 0.50,
        "min_confidence": 0.16,
        "min_actor_fraction": 0.03,
        "max_actor_fraction": 0.38,
    },
    # Lean toward FID/FVD reduction while still avoiding obvious bad geometry.
    "fidlean": {
        "min_coverage": 0.60,
        "min_pose_quality": 0.50,
        "min_confidence": 0.15,
        "min_actor_fraction": 0.00,
        "max_actor_fraction": 0.40,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route BDD test outputs between depth_a75 and actor_depth_a125."
    )
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--actor_dir", required=True)
    parser.add_argument("--diagnostics_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--route_csv", required=True)
    parser.add_argument("--mode", choices=sorted(MODES), default="strict")
    parser.add_argument("--clean", action="store_true")
    return parser


def read_report(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["output_name"]] = row
    return rows


def read_diagnostics(path: Path) -> dict[str, dict[str, float]]:
    accum: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Counter[str] = Counter()
    fields = ["coverage", "pose_quality", "actor_fraction", "mean_confidence", "inliers"]

    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            output_name = row["output_name"]
            counts[output_name] += 1
            for field in fields:
                try:
                    accum[output_name][field] += float(row.get(field) or 0.0)
                except ValueError:
                    pass

    metrics: dict[str, dict[str, float]] = {}
    for output_name, total in accum.items():
        n = max(counts[output_name], 1)
        metrics[output_name] = {field: total.get(field, 0.0) / n for field in fields}
        metrics[output_name]["diagnostic_frames"] = float(n)
    return metrics


def choose_actor(metrics: dict[str, float], mode: str) -> tuple[bool, str]:
    cfg = MODES[mode]
    checks = [
        ("coverage", metrics.get("coverage", 0.0), cfg["min_coverage"], None),
        ("pose_quality", metrics.get("pose_quality", 0.0), cfg["min_pose_quality"], None),
        ("mean_confidence", metrics.get("mean_confidence", 0.0), cfg["min_confidence"], None),
        ("actor_fraction", metrics.get("actor_fraction", 0.0), cfg["min_actor_fraction"], cfg["max_actor_fraction"]),
    ]
    failed: list[str] = []
    for name, value, lower, upper in checks:
        if value < lower:
            failed.append(f"{name}<{lower:g}")
        if upper is not None and value > upper:
            failed.append(f"{name}>{upper:g}")
    if failed:
        return False, ";".join(failed)
    return True, "pass"


def copy_numeric_frames(source_dir: Path, destination: Path) -> int:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    count = 0
    for png in numeric_pngs(source_dir):
        shutil.copy2(png, destination / f"{int(png.stem)}.png")
        count += 1
    return count


def main() -> None:
    args = build_parser().parse_args()

    depth_dir = Path(args.depth_dir).expanduser()
    actor_dir = Path(args.actor_dir).expanduser()
    diagnostics_csv = Path(args.diagnostics_csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    route_csv = Path(args.route_csv).expanduser()

    depth_report = read_report(depth_dir / "report.csv")
    actor_report = read_report(actor_dir / "report.csv")
    diagnostics = read_diagnostics(diagnostics_csv)

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    route_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    bad: list[str] = []
    for output_name in sorted(depth_report):
        depth_row = depth_report[output_name]
        actor_row = actor_report.get(output_name)
        if not actor_row:
            bad.append(f"{output_name}: missing actor report row")
            continue

        target_n = int(depth_row["target_N"])
        metrics = diagnostics.get(output_name, {})
        use_actor, reason = choose_actor(metrics, args.mode) if metrics else (False, "missing_diagnostics")
        source_root = actor_dir if use_actor else depth_dir
        source_choice = "actor_depth_a125" if use_actor else "depth_a75"
        source_dir = source_root / output_name

        verified = verify_sample_output(source_dir, target_n)
        if not verified.ok:
            bad.append(f"{output_name}: {source_choice} verify failed: {verified.notes}")
            continue

        copied = copy_numeric_frames(source_dir, out_dir / output_name)
        out_verified = verify_sample_output(out_dir / output_name, target_n)
        if not out_verified.ok:
            bad.append(f"{output_name}: output verify failed: {out_verified.notes}")
            continue

        rows.append(
            {
                "output_name": output_name,
                "sample_id": depth_row.get("sample_id", ""),
                "target_N": target_n,
                "source_choice": source_choice,
                "reason": reason,
                "frames_copied": copied,
                "coverage": metrics.get("coverage", ""),
                "pose_quality": metrics.get("pose_quality", ""),
                "actor_fraction": metrics.get("actor_fraction", ""),
                "mean_confidence": metrics.get("mean_confidence", ""),
                "inliers": metrics.get("inliers", ""),
                "diagnostic_frames": metrics.get("diagnostic_frames", ""),
                "status": "ok",
            }
        )

    if bad:
        raise RuntimeError("route failed:\n" + "\n".join(bad[:20]))

    fields = [
        "output_name",
        "sample_id",
        "target_N",
        "source_choice",
        "reason",
        "frames_copied",
        "coverage",
        "pose_quality",
        "actor_fraction",
        "mean_confidence",
        "inliers",
        "diagnostic_frames",
        "status",
    ]
    with route_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    counts = Counter(row["source_choice"] for row in rows)
    print(f"mode={args.mode}")
    print(f"out_dir={out_dir}")
    print(f"route_csv={route_csv}")
    print(f"rows={len(rows)}")
    print(f"counts={dict(counts)}")


if __name__ == "__main__":
    main()
