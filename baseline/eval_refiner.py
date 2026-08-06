from __future__ import annotations

import argparse
import json
from pathlib import Path

from .console import kv, section
from .frame_io import import_cv2_np
from .refiner_dataset import RefinerDataset, filter_rows, load_split_rows, sample_rows
from .refiner_features import build_feature_config, refiner_in_channels
from .refiner_model import TinyResidualRefiner, import_torch
from .train_refiner import evaluate, make_loader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--dataset_part", default="")
    parser.add_argument("--view_type", default="")
    parser.add_argument("--phase", default="")
    parser.add_argument("--max_rows", type=int, default=250)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--target_gap_frames", type=int, default=4)
    parser.add_argument("--eval_horizon", type=int, default=4)
    parser.add_argument("--train_width", type=int, default=320)
    parser.add_argument("--train_height", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model_width", type=int, default=48)
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--max_eval_batches", type=int, default=0)

    parser.add_argument("--flow_width", type=int, default=426)
    parser.add_argument("--flow_decay", type=float, default=0.92)
    parser.add_argument("--max_flow_scale", type=float, default=12.0)
    parser.add_argument("--light_flow_decay", type=float, default=0.80)
    parser.add_argument("--light_max_flow_scale", type=float, default=4.0)
    parser.add_argument("--region_motion_tau", type=float, default=1.5)
    parser.add_argument("--region_motion_temp", type=float, default=0.75)
    parser.add_argument("--region_max_alpha", type=float, default=0.85)
    parser.add_argument("--region_overhead_max_alpha", type=float, default=0.45)
    parser.add_argument("--region_mask_blur", type=float, default=3.0)
    parser.add_argument("--random_horizon", action="store_true")
    parser.add_argument("--horizon_min", type=int, default=1)
    parser.add_argument("--horizon_max", type=int, default=16)
    parser.add_argument("--time_normalized_features", action="store_true")
    parser.add_argument("--include_context_features", action="store_true")
    parser.add_argument("--max_context_horizon", type=int, default=120)
    parser.add_argument("--max_context_history_gap", type=int, default=30)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch, _, _ = import_torch()
    cv2, np = import_cv2_np()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint).expanduser()
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    args.model_width = int(ckpt_args.get("model_width", args.model_width))
    args.residual_scale = float(ckpt_args.get("residual_scale", args.residual_scale))
    args.train_width = int(ckpt_args.get("train_width", args.train_width))
    args.train_height = int(ckpt_args.get("train_height", args.train_height))
    args.time_normalized_features = bool(
        ckpt_args.get("time_normalized_features", args.time_normalized_features)
    )
    args.include_context_features = bool(
        ckpt_args.get("include_context_features", args.include_context_features)
    )
    args.max_context_horizon = int(ckpt_args.get("max_context_horizon", args.max_context_horizon))
    args.max_context_history_gap = int(
        ckpt_args.get("max_context_history_gap", args.max_context_history_gap)
    )

    index_dir = Path(args.index_dir).expanduser()
    feature_cfg = build_feature_config(args)
    rows = filter_rows(load_split_rows(index_dir, args.split), args.dataset_part, args.view_type, args.phase)
    rows = sample_rows(rows, args.max_rows, args.seed + 7)
    ds = RefinerDataset(cv2, np, torch, rows, args, feature_cfg, train=False)
    loader = make_loader(torch, ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TinyResidualRefiner.build(
        in_channels=refiner_in_channels(feature_cfg),
        width=args.model_width,
        residual_scale=args.residual_scale,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate(torch, np, cv2, model, loader, device, args)

    section("EVAL RESIDUAL FLOW REFINER")
    kv("checkpoint", checkpoint_path)
    kv("device", device)
    kv("rows", len(rows))
    kv("target_gap_frames", args.target_gap_frames)
    kv("eval_horizon", args.eval_horizon)
    kv("base_psnr", f"{metrics['base_psnr']:.4f}")
    kv("model_psnr", f"{metrics['model_psnr']:.4f}")
    kv("d_psnr", f"{metrics['d_psnr']:+.4f}")
    kv("base_ssim", f"{metrics['base_ssim']:.4f}")
    kv("model_ssim", f"{metrics['model_ssim']:.4f}")
    kv("d_ssim", f"{metrics['d_ssim']:+.4f}")
    kv("base_absdiff", f"{metrics['base_absdiff']:.4f}")
    kv("model_absdiff", f"{metrics['model_absdiff']:.4f}")
    kv("d_absdiff", f"{metrics['d_absdiff']:+.4f}")


if __name__ == "__main__":
    main()
