from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_eval(path: Path, candidate: str):
    if not path.exists():
        return None
    rows = []
    all_rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                all_rows.append(row)
            if row.get("candidate") == candidate and row.get("status") == "ok":
                rows.append(row)
    if not rows:
        return None
    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)
    best_by_sample = {}
    for row in all_rows:
        sample_id = row.get("sample_id", "")
        value = float(row["absdiff"])
        if sample_id not in best_by_sample or value < best_by_sample[sample_id][1]:
            best_by_sample[sample_id] = (row.get("candidate", ""), value)
    wins = sum(1 for name, _ in best_by_sample.values() if name == candidate)
    return len(rows), mean("psnr"), mean("ssim"), mean("absdiff"), wins


def read_proxy(path: Path, candidate: str):
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("candidate") == candidate:
                return float(row["fid_proxy"]), float(row["fvd_proxy"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/experiments/wan_ticft"))
    parser.add_argument("--modes", nargs="+", default=["traffic_detail", "geometry_lock", "copy_bias", "dynamic_geometry"])
    parser.add_argument("--chunk_frames", default="16")
    parser.add_argument("--candidate", default="ltx2b_chunked")
    args = parser.parse_args()

    print(f"{'mode':<18} {'N':>4} {'PSNR':>9} {'SSIM':>8} {'AbsDiff':>9} {'Wins':>5} {'FID':>9} {'FVD':>9}")
    print("-" * 78)
    for mode in args.modes:
        out = args.root / f"ltx2b_chunked_bdd24_prompt_{mode}_c{args.chunk_frames}"
        ev = read_eval(out / "eval_pixel.csv", args.candidate)
        pr = read_proxy(out / "eval_proxy.csv", args.candidate)
        if ev is None:
            print(f"{mode:<18} {'missing':>4}")
            continue
        n, p, s, d, wins = ev
        fid, fvd = pr if pr else ("", "")
        fid_text = f"{fid:.4f}" if isinstance(fid, float) else ""
        fvd_text = f"{fvd:.4f}" if isinstance(fvd, float) else ""
        print(f"{mode:<18} {n:>4} {p:>9.4f} {s:>8.4f} {d:>9.4f} {wins:>5} {fid_text:>9} {fvd_text:>9}")


if __name__ == "__main__":
    main()
