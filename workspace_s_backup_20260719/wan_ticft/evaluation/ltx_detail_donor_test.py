from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs
from wan_ticft.evaluation.ltx_detail_donor import all_frames, resize_like, transfer_detail


def read_donor_report(path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            video = row.get("video") or ""
            if row.get("status") != "ok" or not video:
                continue
            for key in ("output_name", "sample_id"):
                value = row.get(key) or ""
                if value:
                    out[value] = Path(video)
    return out


def read_bdd_rows(index_csv: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with index_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("view_type") == "bdd":
                rows.append(row)
                if limit and len(rows) >= limit:
                    break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply LTX high-frequency detail donor to BDD test frame folders.")
    parser.add_argument("--index_csv", type=Path, default=Path("/workspace/experiments/index/test_index.csv"))
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--donor_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--sigma", type=float, default=2.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, _np = import_cv2_np()
    rows = read_bdd_rows(args.index_csv, args.limit)
    donor_report = read_donor_report(args.donor_report)
    if args.clean and args.out_dir.exists():
        import shutil

        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report: list[dict[str, str]] = []
    for pos, row in enumerate(rows, start=1):
        output_name = row["source_folder"] or row["sample_id"]
        sample_id = row["sample_id"]
        target_n = int(row.get("target_N") or 0)
        base_paths = numeric_pngs(args.base_dir / output_name)
        donor_path = donor_report.get(output_name) or donor_report.get(sample_id)
        out_sample = args.out_dir / output_name
        status = "ok"
        notes = ""
        try:
            if not base_paths:
                raise RuntimeError("missing base frames")
            if donor_path is None:
                raise RuntimeError("missing donor video")
            if target_n and len(base_paths) != target_n:
                notes = f"base frames {len(base_paths)} != target_N {target_n}"
            donor_frames = all_frames(cv2, donor_path, len(base_paths))
            out_sample.mkdir(parents=True, exist_ok=True)
            for idx, base_path in enumerate(base_paths):
                base_bgr = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
                if base_bgr is None:
                    raise RuntimeError(f"cannot read {base_path}")
                base = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
                donor = resize_like(cv2, donor_frames[idx], base)
                out = transfer_detail(cv2, base, [donor], alpha=args.alpha, sigma=args.sigma)
                cv2.imwrite(str(out_sample / f"{int(base_path.stem)}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        except Exception as exc:  # noqa: BLE001
            status = "bad"
            notes = str(exc)
        report.append(
            {
                "output_name": output_name,
                "sample_id": sample_id,
                "target_N": str(target_n),
                "status": status,
                "notes": notes,
            }
        )
        print(f"[{pos}/{len(rows)}] {output_name} | {status} {notes}", flush=True)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output_name", "sample_id", "target_N", "status", "notes"])
        writer.writeheader()
        writer.writerows(report)
    bad = [row for row in report if row["status"] != "ok"]
    print(f"output={args.out_dir}")
    print(f"report={report_path}")
    print(f"bad={len(bad)}/{len(report)}")
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
