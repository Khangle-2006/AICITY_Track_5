from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path("/workspace/experiments/wan_ticft")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_bdd_rows(path: Path, count: int) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        if row.get("view_type") == "bdd":
            rows.append(row)
            if len(rows) == count:
                break
    return rows


def main() -> None:
    out = ROOT / "ltxv_bdd_dense6_eval"
    out.mkdir(parents=True, exist_ok=True)

    val2_manifest = ROOT / "ltxv_bdd_val2_gate" / "selected_manifest.jsonl"
    if val2_manifest.exists():
        rows = read_jsonl(val2_manifest)
    else:
        rows = first_bdd_rows(ROOT / "manifests" / "val_ticft_balanced_125_frames.jsonl", 2)
    rows += read_jsonl(ROOT / "ltxv_bdd_mixed4_gate" / "selected_manifest.jsonl")
    with (out / "source_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    reports = []
    for report_path in [
        ROOT / "ltxv_bdd_val2_gate" / "report.csv",
        ROOT / "ltxv_bdd_mixed4_gate" / "report.csv",
    ]:
        with report_path.open(encoding="utf-8", newline="") as f:
            reports.extend(csv.DictReader(f))
    fields = list(reports[0])
    with (out / "ltx_report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reports)

    print(f"rows {len(rows)}")
    print(f"out {out}")


if __name__ == "__main__":
    main()
