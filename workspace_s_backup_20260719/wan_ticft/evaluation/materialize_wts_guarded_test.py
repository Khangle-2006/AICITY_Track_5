from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import numeric_pngs
from baseline.verify import verify_sample_output


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_notes(notes: str) -> dict[str, float]:
    values = {}
    for chunk in notes.replace(";", ",").split(","):
        if "=" not in chunk:
            continue
        key, raw = chunk.strip().split("=", 1)
        try:
            values[key.strip()] = float(raw)
        except ValueError:
            continue
    return values


def passes_guard(stats: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        stats.get("mask_mean", 0.0) <= args.max_mean
        and stats.get("mask_p90", 0.0) <= args.max_p90
        and stats.get("motion_p90", 0.0) <= args.max_motion_p90
    )


def copy_sample(src: Path, dst: Path) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for png in numeric_pngs(src):
        shutil.copy2(png, dst / f"{int(png.stem)}.png")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build guarded WTS test outputs from current and actor-layer dirs.")
    parser.add_argument("--index_csv", type=Path, required=True)
    parser.add_argument("--current_dir", type=Path, required=True)
    parser.add_argument("--actor_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_mean", type=float, default=0.018)
    parser.add_argument("--max_p90", type=float, default=0.25)
    parser.add_argument("--max_motion_p90", type=float, default=6.0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    current_manifest = read_manifest(args.current_dir / "manifest.csv")
    actor_manifest = read_manifest(args.actor_dir / "manifest.csv")
    current_by_name = {Path(row["output_dir"]).name: row for row in current_manifest if row.get("status") == "ok"}
    actor_by_name = {Path(row["output_dir"]).name: row for row in actor_manifest if row.get("status") == "ok"}
    index_rows = read_manifest(args.index_csv)

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    selected = 0
    for row in index_rows:
        if row.get("view_type") not in {"wts_overhead_or_fixed_camera", "wts_vehicle_or_ip_camera"}:
            continue
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        use_actor = False
        if row.get("view_type") == "wts_overhead_or_fixed_camera":
            actor_row = actor_by_name.get(output_name)
            if actor_row is not None and passes_guard(parse_notes(actor_row.get("notes", "")), args):
                use_actor = True
        src_root = args.actor_dir if use_actor else args.current_dir
        src = src_root / output_name
        dst = args.out_dir / output_name
        frames = copy_sample(src, dst)
        verified = verify_sample_output(dst, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verify failed after copy: {verified.notes}")
        if use_actor:
            selected += 1
        report.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "view_type": row["view_type"],
                "target_N": target_n,
                "frames_found": frames,
                "use_actor": int(use_actor),
                "source": "actor_layer_small_filled" if use_actor else "current",
                "status": "ok",
            }
        )
        print(f"{output_name} | {'actor' if use_actor else 'current'}", flush=True)

    report_path = args.out_dir / "guarded_report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    (args.out_dir / "summary.json").write_text(
        json.dumps({"rows": len(report), "selected_actor": selected, "status": "ok"}, indent=2),
        encoding="utf-8",
    )
    print(f"selected_actor={selected}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
