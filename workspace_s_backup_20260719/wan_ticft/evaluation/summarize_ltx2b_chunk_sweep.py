from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_candidate(path: Path, candidate: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Candidate") == candidate or row.get("candidate") == candidate:
                return row
    return None


def read_eval_summary(path: Path, candidate: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("candidate") == candidate and row.get("status") == "ok":
                rows.append(row)
    if not rows:
        return None
    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)
    wins = sum(1 for row in rows if row.get("wins") == "1")
    return {
        "N": str(len(rows)),
        "PSNR": f"{mean('psnr'):.4f}",
        "SSIM": f"{mean('ssim'):.4f}",
        "AbsDiff": f"{mean('absdiff'):.4f}",
        "Wins": str(wins),
    }


def read_proxy_summary(path: Path, candidate: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        name = row.get("candidate") or row.get("Candidate") or row.get("name")
        if name == candidate:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/experiments/wan_ticft"))
    parser.add_argument("--sizes", nargs="+", default=["8", "12", "16", "24"])
    parser.add_argument("--candidate", default="ltx2b_chunked")
    args = parser.parse_args()

    print(f"{'chunk':>6} {'N':>4} {'PSNR':>9} {'SSIM':>8} {'AbsDiff':>9} {'Wins':>5} {'FID':>9} {'FVD':>9}")
    print("-" * 72)
    for size in args.sizes:
        safe_size = size.replace(".", "p")
        out = args.root / f"ltx2b_chunked_traffic_bdd24_c{safe_size}"
        pixel = read_eval_summary(out / "eval_pixel.csv", args.candidate)
        proxy = read_proxy_summary(out / "eval_proxy.csv", args.candidate)
        fid = fvd = ""
        if proxy:
            fid = proxy.get("fid_proxy") or proxy.get("FID proxy") or proxy.get("fid") or ""
            fvd = proxy.get("fvd_proxy") or proxy.get("FVD proxy") or proxy.get("fvd") or ""
        if not pixel:
            print(f"{size:>6} {'missing':>4}")
            continue
        print(
            f"{size:>6} {pixel['N']:>4} {pixel['PSNR']:>9} {pixel['SSIM']:>8} "
            f"{pixel['AbsDiff']:>9} {pixel['Wins']:>5} {fid:>9} {fvd:>9}"
        )


if __name__ == "__main__":
    main()
