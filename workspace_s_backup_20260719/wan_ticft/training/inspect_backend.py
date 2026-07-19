"""Inspect local Wan/diffusers training backend availability.

This script is intentionally dependency-light. Run it on the training server
before wiring the real TIC-FT backend so we know which API/checkpoints exist.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
from pathlib import Path


MODULES = [
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "safetensors",
    "wan",
    "wan2_1",
    "wan_t2v",
    "wan_i2v",
]

DIR_PATTERNS = [
    "*Wan*",
    "*wan*",
    "*diffusers*",
    "*CogVideo*",
    "*cogvideo*",
]

MODEL_FILE_SUFFIXES = {
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
}


def print_kv(key: str, value: object, width: int = 42) -> None:
    print(f"{key:<{width}} {value}")


def run_text(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=15)
    except Exception as exc:  # pragma: no cover - environment probe
        return f"(failed: {exc})"
    return out.strip()


def module_status() -> None:
    print("\nMODULES")
    print("-" * 80)
    for name in MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None:
            print_kv(name, "NO")
            continue
        origin = spec.origin or "(namespace/package)"
        print_kv(name, f"OK | {origin}")


def torch_status() -> None:
    print("\nTORCH / GPU")
    print("-" * 80)
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment probe
        print_kv("torch import", f"failed: {exc}")
        return

    print_kv("torch version", torch.__version__)
    print_kv("cuda available", torch.cuda.is_available())
    print_kv("cuda device count", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        gb = props.total_memory / (1024**3)
        print_kv(f"cuda:{idx}", f"{props.name} | {gb:.1f} GB")

    print("\nNVIDIA-SMI")
    print("-" * 80)
    print(run_text(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"]))


def path_candidates(roots: list[Path], max_depth: int) -> None:
    print("\nCANDIDATE DIRECTORIES")
    print("-" * 80)
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for pattern in DIR_PATTERNS:
            for path in root.rglob(pattern):
                if not path.is_dir():
                    continue
                try:
                    rel_depth = len(path.relative_to(root).parts)
                except ValueError:
                    rel_depth = 999
                if rel_depth > max_depth:
                    continue
                if path.name.lower().startswith("video"):
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                print(resolved)
    if not seen:
        print("(none)")


def checkpoint_candidates(roots: list[Path], max_depth: int) -> None:
    print("\nCANDIDATE MODEL FILES")
    print("-" * 80)
    found = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MODEL_FILE_SUFFIXES:
                continue
            try:
                rel_depth = len(path.relative_to(root).parts)
            except ValueError:
                rel_depth = 999
            if rel_depth > max_depth:
                continue
            size_gb = path.stat().st_size / (1024**3)
            if size_gb < 0.05:
                continue
            print(f"{path.resolve()} | {size_gb:.2f} GB")
            found += 1
            if found >= 80:
                print("(truncated)")
                return
    if found == 0:
        print("(none)")


def env_status() -> None:
    print("\nENV PATHS")
    print("-" * 80)
    for key in [
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "DIFFUSERS_CACHE",
    ]:
        print_kv(key, os.environ.get(key, "(unset)"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["/workspace", "/workspace/models", "/workspace/checkpoints", "/root/.cache/huggingface"],
        help="Roots to scan for Wan/diffusers repos and checkpoint files.",
    )
    parser.add_argument("--max_depth", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(p).expanduser() for p in args.roots]

    print("=" * 100)
    print("INSPECT WAN TIC-FT BACKEND")
    print("=" * 100)
    print_kv("roots", ", ".join(str(p) for p in roots))
    print_kv("max_depth", args.max_depth)

    env_status()
    module_status()
    torch_status()
    path_candidates(roots, args.max_depth)
    checkpoint_candidates(roots, args.max_depth)


if __name__ == "__main__":
    main()
