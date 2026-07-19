from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .layout import describe_layout
from .manifest_dataset import ManifestDataset, summarize_manifest


def section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def kv(key: str, value: Any) -> None:
    print(f"{key:<42} {value}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wan/TIC-FT training entrypoint skeleton.")
    parser.add_argument("--train_jsonl", required=True, type=Path)
    parser.add_argument("--val_jsonl", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--backend", choices=["dryrun", "wan"], default="dryrun")
    parser.add_argument("--wan_repo", type=Path, default=None)
    parser.add_argument("--wan_model", default="")
    parser.add_argument("--adapter", choices=["none", "lora", "spatial_temporal"], default="lora")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--max_timestep", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--strict", action="store_true", help="Fail if any manifest row is invalid.")
    return parser


def dryrun(args, train_ds: ManifestDataset, val_ds: ManifestDataset) -> None:
    train_summary = summarize_manifest(train_ds.rows)
    val_summary = summarize_manifest(val_ds.rows)
    first_layout = describe_layout(train_ds.rows[0], args.max_timestep) if train_ds.rows else {}

    plan = {
        "backend": args.backend,
        "adapter": args.adapter,
        "lora_rank": args.lora_rank,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "wan_repo": str(args.wan_repo) if args.wan_repo else "",
        "wan_model": args.wan_model,
        "ticft_layout_example": first_layout,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "notes": [
            "This dry-run validates Track 5 TIC-FT manifests.",
            "Wan backend should encode condition frames, create noised condition buffers, create noisy target latents, and compute loss only on target slice.",
            "Keep Wan backbone frozen for first experiments; train LoRA/adapters only.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "train_plan.json", plan)
    args_json = vars(args).copy()
    args_json["train_jsonl"] = str(args.train_jsonl)
    args_json["val_jsonl"] = str(args.val_jsonl)
    args_json["out_dir"] = str(args.out_dir)
    args_json["wan_repo"] = str(args.wan_repo) if args.wan_repo else ""
    write_json(args.out_dir / "args.json", args_json)

    section("DRYRUN SUMMARY")
    kv("train rows", train_summary["rows"])
    kv("val rows", val_summary["rows"])
    kv("train layouts", train_summary["layouts"])
    kv("train by_view", train_summary["by_view"])
    kv("val by_view", val_summary["by_view"])
    kv("prompt words train mean", f"{train_summary['prompt_words_mean']:.1f}")
    kv("prompt words val mean", f"{val_summary['prompt_words_mean']:.1f}")
    kv("layout example", first_layout)
    kv("plan", args.out_dir / "train_plan.json")


def main() -> None:
    args = build_parser().parse_args()

    section("WAN TIC-FT TRAINING")
    kv("train_jsonl", args.train_jsonl)
    kv("val_jsonl", args.val_jsonl)
    kv("out_dir", args.out_dir)
    kv("backend", args.backend)
    kv("adapter", args.adapter)

    train_ds = ManifestDataset(args.train_jsonl, require_target=True)
    val_ds = ManifestDataset(args.val_jsonl, require_target=True)

    train_bad = train_ds.validation_errors()
    val_bad = val_ds.validation_errors()
    kv("train bad rows", len(train_bad))
    kv("val bad rows", len(val_bad))
    if train_bad or val_bad:
        section("FIRST MANIFEST ERRORS")
        for name, errors in (train_bad + val_bad)[:20]:
            print(f"  - {name}: {'; '.join(errors[:3])}")
        if args.strict:
            raise SystemExit(2)

    if args.backend == "dryrun":
        dryrun(args, train_ds, val_ds)
        return

    if args.wan_repo is None or not args.wan_repo.exists():
        raise SystemExit("--backend wan requires --wan_repo pointing to the installed Wan training repo")
    raise SystemExit(
        "Wan backend hook is not implemented yet. Dry-run is complete; next step is wiring this manifest/layout into the local Wan training code."
    )


if __name__ == "__main__":
    main()

