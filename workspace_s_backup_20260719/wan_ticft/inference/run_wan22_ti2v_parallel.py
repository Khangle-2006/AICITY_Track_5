"""Launch Wan2.2 TI2V manifest inference shards across multiple GPUs.

This uses the safe subprocess runner (``run_wan22_ti2v.py``) per shard. Each
worker keeps only one Wan generation active on its assigned GPU, which avoids
the resident-model OOM we hit with the direct fast runner on 40GB A100s.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_kv(key: str, value: object, width: int = 42) -> None:
    print(f"{key:<{width}} {value}")


def count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def split_counts(total: int, workers: int) -> list[tuple[int, int]]:
    base = total // workers
    rem = total % workers
    shards: list[tuple[int, int]] = []
    cursor = 0
    for i in range(workers):
        count = base + (1 if i < rem else 0)
        shards.append((cursor, count))
        cursor += count
    return [(start, count) for start, count in shards if count > 0]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_reports(out_dir: Path, shard_dirs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for shard_idx, shard_dir in enumerate(shard_dirs):
        for row in read_csv(shard_dir / "report.csv"):
            row = dict(row)
            row["shard"] = str(shard_idx)
            rows.append(row)
    if not rows:
        return []

    keys = ["shard"] + [k for k in rows[0].keys() if k != "shard"]
    with (out_dir / "report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    return rows


def write_summary(out_dir: Path, rows: list[dict[str, str]], args: argparse.Namespace, elapsed: float) -> None:
    by_status: dict[str, int] = {}
    by_view: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "(missing)")
        by_status[status] = by_status.get(status, 0) + 1
        view = row.get("view_type", "(unknown)")
        by_view[view] = by_view.get(view, 0) + 1
    summary = {
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "gpus": args.gpus,
        "rows": len(rows),
        "elapsed_sec": elapsed,
        "by_status": by_status,
        "by_view": by_view,
        "sample_steps": args.sample_steps,
        "frame_num": args.frame_num,
        "size": args.size,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--wan_repo", type=Path, default=Path("/workspace/third_party/Wan2.2"))
    parser.add_argument("--ckpt_dir", type=Path, default=Path("/workspace/models/Wan2.2-TI2V-5B"))
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU ids.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--sample_solver", default="unipc")
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--base_seed", type=int, default=11)
    parser.add_argument("--image_policy", choices=["first", "middle", "last"], default="last")
    parser.add_argument("--max_prompt_words", type=int, default=120)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    args.wan_repo = args.wan_repo.resolve()
    args.ckpt_dir = args.ckpt_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")

    available = max(0, count_jsonl(args.manifest) - args.offset)
    total = min(available, args.limit) if args.limit is not None else available
    shards = split_counts(total, len(gpus))

    print_header("WAN2.2 TI2V PARALLEL MANIFEST INFERENCE")
    print_kv("manifest", args.manifest)
    print_kv("out_dir", args.out_dir)
    print_kv("gpus", ",".join(gpus))
    print_kv("offset", args.offset)
    print_kv("rows", total)
    print_kv("workers", len(shards))
    print_kv("sample_steps", args.sample_steps)

    procs: list[tuple[int, str, Path, subprocess.Popen[str]]] = []
    shard_dirs: list[Path] = []
    start_time = time.time()
    for shard_idx, (local_start, count) in enumerate(shards):
        gpu = gpus[shard_idx % len(gpus)]
        shard_offset = args.offset + local_start
        shard_dir = args.out_dir / f"shard_{shard_idx:02d}_gpu{gpu}"
        shard_dirs.append(shard_dir)
        log_path = args.out_dir / f"shard_{shard_idx:02d}_gpu{gpu}.log"
        cmd = [
            sys.executable,
            "-m",
            "wan_ticft.inference.run_wan22_ti2v",
            "--manifest",
            str(args.manifest),
            "--out_dir",
            str(shard_dir),
            "--wan_repo",
            str(args.wan_repo),
            "--ckpt_dir",
            str(args.ckpt_dir),
            "--offset",
            str(shard_offset),
            "--limit",
            str(count),
            "--sample_steps",
            str(args.sample_steps),
            "--frame_num",
            str(args.frame_num),
            "--size",
            args.size,
            "--sample_solver",
            args.sample_solver,
            "--sample_shift",
            str(args.sample_shift),
            "--sample_guide_scale",
            str(args.sample_guide_scale),
            "--base_seed",
            str(args.base_seed + shard_offset),
            "--image_policy",
            args.image_policy,
            "--max_prompt_words",
            str(args.max_prompt_words),
            "--cuda_visible_devices",
            gpu,
        ]
        if args.skip_existing:
            cmd.append("--skip_existing")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log = log_path.open("w", encoding="utf-8")
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd="/workspace", env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        procs.append((shard_idx, gpu, log_path, proc))
        print_kv(f"launched shard {shard_idx}", f"gpu={gpu}, offset={shard_offset}, count={count}, log={log_path}")

    failures = 0
    for shard_idx, gpu, log_path, proc in procs:
        code = proc.wait()
        print_kv(f"finished shard {shard_idx}", f"gpu={gpu}, returncode={code}, log={log_path}")
        if code != 0:
            failures += 1

    elapsed = time.time() - start_time
    rows = merge_reports(args.out_dir, shard_dirs)
    write_summary(args.out_dir, rows, args, elapsed)
    failed_samples = sum(row.get("status") == "failed" for row in rows)

    print_header("RESULT")
    print_kv("elapsed_sec", f"{elapsed:.2f}")
    print_kv("shard_failures", failures)
    print_kv("failed_samples", failed_samples)
    print_kv("merged report", args.out_dir / "report.csv")
    print_kv("summary", args.out_dir / "summary.json")

    if failures or failed_samples:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
