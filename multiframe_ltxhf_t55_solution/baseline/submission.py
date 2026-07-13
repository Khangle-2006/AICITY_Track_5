from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .console import kv, section
from .frame_io import numeric_pngs
from .index_io import load_test_rows
from .naming import parse_int, resolve_output_name
from .verify import verify_sample_output


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_expected_outputs(index_dir: Path) -> list[dict[str, Any]]:
    rows = load_test_rows(index_dir)
    used: set[str] = set()
    out = []
    for row in rows:
        output_name = resolve_output_name(row, used)
        out.append(
            {
                "output_name": output_name,
                "sample_id": row.get("sample_id", ""),
                "source_folder": row.get("source_folder", ""),
                "target_N": parse_int(row.get("target_N", ""), 0),
                "view_type": row.get("view_type", ""),
                "dataset_part": row.get("dataset_part", ""),
            }
        )
    return out


def verify_generation(index_dir: Path, gen_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = build_expected_outputs(index_dir)
    results = []
    bad = []
    for item in expected:
        out_dir = gen_dir / item["output_name"]
        vr = verify_sample_output(out_dir, item["target_N"])
        row = {
            **item,
            "output_dir": str(out_dir),
            "frames_found": vr.frames_found,
            "status": "ok" if vr.ok else "bad",
            "notes": vr.notes,
        }
        results.append(row)
        if not vr.ok:
            bad.append(row)
    return results, bad


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "output_name",
        "sample_id",
        "source_folder",
        "dataset_part",
        "view_type",
        "target_N",
        "output_dir",
        "frames_found",
        "status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def prepare_submission(results: list[dict[str, Any]], gen_dir: Path, submission_dir: Path, clean: bool) -> None:
    if clean and submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    for item in results:
        if item["status"] != "ok":
            continue
        src_dir = gen_dir / item["output_name"]
        dst_dir = submission_dir / item["output_name"]
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for png in numeric_pngs(src_dir):
            shutil.copy2(png, dst_dir / png.name)


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*.png")):
            zf.write(path, path.relative_to(src_dir))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_view: dict[str, int] = {}
    frames = 0
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        by_view[row["view_type"]] = by_view.get(row["view_type"], 0) + 1
        frames += parse_int(str(row.get("frames_found", "")), 0)
    return {"rows": len(rows), "by_status": by_status, "by_view": by_view, "frames_found": frames}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--gen_dir", required=True, help="Generated output root from baseline_end_to_end.py")
    parser.add_argument("--submission_dir", required=True, help="Clean output folder to create")
    parser.add_argument("--report_csv", default="")
    parser.add_argument("--zip", default="", help="Optional zip path")
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no_clean", action="store_false", dest="clean")
    parser.add_argument("--verify_only", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    index_dir = Path(args.index_dir).expanduser()
    gen_dir = Path(args.gen_dir).expanduser()
    submission_dir = Path(args.submission_dir).expanduser()
    report_csv = Path(args.report_csv).expanduser() if args.report_csv else submission_dir.parent / "submission_verify.csv"
    zip_path = Path(args.zip).expanduser() if args.zip else None

    section("TRACK 5 SUBMISSION PREP")
    kv("index_dir", index_dir)
    kv("gen_dir", gen_dir)
    kv("submission_dir", submission_dir)
    kv("report_csv", report_csv)
    kv("zip", zip_path or "(none)")
    kv("verify_only", args.verify_only)

    results, bad = verify_generation(index_dir, gen_dir)
    write_csv(report_csv, results)
    summary = summarize(results)

    if bad:
        section("VERIFY FAILED")
        kv("bad samples", len(bad))
        for item in bad[:20]:
            print(f"  - {item['output_name']} | found={item['frames_found']} expected={item['target_N']} | {item['notes']}")
        raise SystemExit(2)

    if not args.verify_only:
        prepare_submission(results, gen_dir, submission_dir, clean=args.clean)
        if zip_path:
            zip_dir(submission_dir, zip_path)

    section("RESULT")
    kv("rows", summary["rows"])
    kv("frames_found", summary["frames_found"])
    kv("by_status", summary["by_status"])
    kv("by_view", summary["by_view"])
    kv("report_csv", report_csv)
    if not args.verify_only:
        kv("submission_dir", submission_dir)
    if zip_path:
        kv("zip", zip_path)


if __name__ == "__main__":
    main()
