from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from baseline.console import kv, section
from baseline.frame_io import numeric_pngs
from baseline.submission import build_expected_outputs, zip_dir
from baseline.verify import verify_sample_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a submission by selecting Wan or region-flow outputs per view."
    )
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--wan_dir", required=True)
    parser.add_argument("--region_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--report_csv", required=True)
    parser.add_argument("--zip", default="")
    parser.add_argument("--region_views", nargs="+", required=True)
    parser.add_argument("--wan_label", default="wan_zero_shot")
    parser.add_argument("--region_label", default="region_norm_medium")
    parser.add_argument(
        "--view_override",
        action="append",
        default=[],
        metavar="VIEW=DIR=LABEL",
        help="Override one view with a dedicated source directory and label.",
    )
    parser.add_argument(
        "--source_folder_override",
        action="append",
        default=[],
        metavar="VIEW=SUBSTRING=DIR=LABEL",
        help=(
            "Override samples from one view whose source_folder contains SUBSTRING. "
            "This takes precedence over --view_override."
        ),
    )
    parser.add_argument("--clean", action="store_true")
    return parser


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "output_name",
        "sample_id",
        "source_folder",
        "dataset_part",
        "view_type",
        "target_N",
        "source_choice",
        "source_dir",
        "frames_found",
        "status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = build_parser().parse_args()

    index_dir = Path(args.index_dir).expanduser()
    wan_dir = Path(args.wan_dir).expanduser()
    region_dir = Path(args.region_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    report_csv = Path(args.report_csv).expanduser()
    zip_path = Path(args.zip).expanduser() if args.zip else None
    region_views = set(args.region_views)
    view_overrides: dict[str, tuple[Path, str]] = {}
    for item in args.view_override:
        parts = item.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"invalid --view_override {item!r}; expected VIEW=DIR=LABEL")
        view, directory, label = parts
        view_overrides[view] = (Path(directory).expanduser(), label)
    source_folder_overrides: list[tuple[str, str, Path, str]] = []
    for item in args.source_folder_override:
        parts = item.split("=", 3)
        if len(parts) != 4:
            raise ValueError(
                f"invalid --source_folder_override {item!r}; expected VIEW=SUBSTRING=DIR=LABEL"
            )
        view, substring, directory, label = parts
        source_folder_overrides.append((view, substring, Path(directory).expanduser(), label))

    section("BUILD HYBRID WAN + REGION SUBMISSION")
    kv("index_dir", index_dir)
    kv("wan_dir", wan_dir)
    kv("region_dir", region_dir)
    kv("out_dir", out_dir)
    kv("region_views", sorted(region_views))
    kv("view_overrides", {view: str(root) for view, (root, _) in view_overrides.items()})
    kv(
        "source_folder_overrides",
        [
            {"view": view, "substring": substring, "root": str(root), "label": label}
            for view, substring, root, label in source_folder_overrides
        ],
    )
    kv("zip", zip_path or "(none)")

    expected = build_expected_outputs(index_dir)
    plan: list[dict[str, object]] = []
    bad: list[dict[str, object]] = []

    for item in expected:
        matched_override = None
        for view, substring, root, label in source_folder_overrides:
            if item["view_type"] == view and substring in str(item.get("source_folder", "")):
                matched_override = (root, label)
                break
        if matched_override is not None:
            source_root, source_choice = matched_override
        elif item["view_type"] in view_overrides:
            source_root, source_choice = view_overrides[item["view_type"]]
        else:
            use_region = item["view_type"] in region_views
            source_choice = args.region_label if use_region else args.wan_label
            source_root = region_dir if use_region else wan_dir
        source_dir = source_root / str(item["output_name"])
        verified = verify_sample_output(source_dir, int(item["target_N"]))
        row = {
            **item,
            "source_choice": source_choice,
            "source_dir": str(source_dir),
            "frames_found": verified.frames_found,
            "status": "ok" if verified.ok else "bad",
            "notes": verified.notes,
        }
        plan.append(row)
        if not verified.ok:
            bad.append(row)

    write_report(report_csv, plan)
    if bad:
        section("SOURCE VERIFY FAILED")
        for row in bad[:20]:
            print(
                f"  - {row['output_name']} | {row['source_choice']} | "
                f"found={row['frames_found']} expected={row['target_N']} | {row['notes']}"
            )
        raise SystemExit(2)

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in plan:
        source_dir = Path(str(row["source_dir"]))
        destination = out_dir / str(row["output_name"])
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        for png in numeric_pngs(source_dir):
            shutil.copy2(png, destination / f"{int(png.stem)}.png")

        verified = verify_sample_output(destination, int(row["target_N"]))
        if not verified.ok:
            raise RuntimeError(
                f"Hybrid output verification failed for {row['output_name']}: {verified.notes}"
            )

    if zip_path:
        zip_dir(out_dir, zip_path)

    by_source = Counter(str(row["source_choice"]) for row in plan)
    by_view = Counter(str(row["view_type"]) for row in plan)
    frames_found = sum(int(row["frames_found"]) for row in plan)

    section("RESULT")
    kv("rows", len(plan))
    kv("frames_found", frames_found)
    kv("by_source", dict(by_source))
    kv("by_view", dict(by_view))
    kv("report_csv", report_csv)
    kv("submission_dir", out_dir)
    if zip_path:
        kv("zip", zip_path)


if __name__ == "__main__":
    main()
