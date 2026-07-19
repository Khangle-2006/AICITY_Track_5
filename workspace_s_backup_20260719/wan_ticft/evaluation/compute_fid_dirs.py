"""Compute pytorch-fid between two already-materialized image directories."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    from pytorch_fid.fid_score import calculate_fid_given_paths
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    value = calculate_fid_given_paths(
        [str(args.real_dir), str(args.candidate_dir)],
        args.batch_size,
        device,
        2048,
        2,
    )
    print(f"FID={value:.6f}")


if __name__ == "__main__":
    main()
