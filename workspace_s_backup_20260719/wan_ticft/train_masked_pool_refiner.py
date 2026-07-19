"""Masked temporal refiner trained from BDD candidate pool to GT.

This is a small smoke experiment for the Track 5 temporal ghosting issue.
It learns a residual from candidate future frames to real GT, emphasizing
motion/disagreement regions while preserving the candidate outside the mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def sorted_frames(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)


def read_video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames


def motion_mask(prev: np.ndarray, curr: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_next = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY).astype(np.float32)
    raw = np.maximum(np.abs(gray_curr - gray_prev), np.abs(gray_next - gray_curr)) / 255.0
    mask = np.clip((raw - 0.018) / 0.16, 0.0, 1.0)
    return cv2.GaussianBlur(mask, (9, 9), 1.5)


def disagreement_mask(a: np.ndarray, b: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
    diff = cv2.cvtColor(cv2.absdiff(a, b), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if c is not None:
        diff = np.maximum(diff, cv2.cvtColor(cv2.absdiff(a, c), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
    mask = np.clip((diff - 0.015) / 0.12, 0.0, 1.0)
    return cv2.GaussianBlur(mask, (9, 9), 1.2)


def make_input_stack(
    medium_frames: list[np.ndarray],
    long_frames: list[np.ndarray] | None,
    actor_frames: list[np.ndarray] | None,
    idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(medium_frames)
    prev_i = max(0, idx - 1)
    next_i = min(n - 1, idx + 1)
    prev_frame = medium_frames[prev_i]
    curr_frame = medium_frames[idx]
    next_frame = medium_frames[next_i]
    long_frame = long_frames[idx] if long_frames is not None else curr_frame
    actor_frame = actor_frames[idx] if actor_frames is not None else curr_frame
    infer = np.maximum(motion_mask(prev_frame, curr_frame, next_frame), disagreement_mask(curr_frame, long_frame, actor_frame))
    stack = np.concatenate([prev_frame, curr_frame, next_frame, long_frame, actor_frame, (infer[..., None] * 255.0).astype(np.uint8)], axis=2)
    return stack, curr_frame, infer


def build_cache(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    root = args.pool_root
    medium_paths = sorted((root / "region_norm_medium").glob("*.mp4"))
    if not medium_paths:
        raise RuntimeError(f"no mp4s in {root / 'region_norm_medium'}")
    names = [p.name for p in medium_paths]
    if args.clean and args.cache_dir.exists():
        shutil.rmtree(args.cache_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    x_arr = np.empty((args.samples, args.patch, args.patch, 16), dtype=np.uint8)
    target_arr = np.empty((args.samples, args.patch, args.patch, 3), dtype=np.uint8)
    center_arr = np.empty((args.samples, args.patch, args.patch, 3), dtype=np.uint8)
    mask_arr = np.empty((args.samples, args.patch, args.patch), dtype=np.uint8)
    made = 0
    tries = 0
    while made < args.samples:
        tries += 1
        if tries > args.samples * 30:
            raise RuntimeError(f"only cached {made}/{args.samples} after {tries} attempts")
        name = rng.choice(names)
        try:
            medium = read_video_frames(root / "region_norm_medium" / name)
            target = read_video_frames(root / "target" / name)
            long = read_video_frames(root / "region_norm_long24" / name)
            actor = read_video_frames(root / "actor_layer_small_filled" / name)
        except RuntimeError:
            continue
        count = min(len(medium), len(target), len(long), len(actor))
        if count < 3:
            continue
        idx = rng.randrange(1, count - 1)
        h, w = medium[0].shape[:2]
        patch = args.patch
        if h < patch or w < patch:
            raise RuntimeError(f"patch {patch} larger than video frame {w}x{h}")
        x0 = rng.randrange(0, w - patch + 1)
        y0 = rng.randrange(0, h - patch + 1)
        stack, center, infer = make_input_stack(medium[:count], long[:count], actor[:count], idx)
        tgt = target[idx]
        gt_motion = motion_mask(target[max(0, idx - 1)], tgt, target[min(count - 1, idx + 1)])
        err = cv2.cvtColor(cv2.absdiff(center, tgt), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        loss_mask = np.maximum.reduce([infer, gt_motion, np.clip((err - 0.025) / 0.12, 0.0, 1.0)])
        loss_mask = cv2.GaussianBlur(loss_mask, (11, 11), 1.8)
        crop = np.s_[y0 : y0 + patch, x0 : x0 + patch]
        if rng.random() < 0.5:
            x_arr[made] = cv2.flip(stack[crop], 1)
            target_arr[made] = cv2.flip(tgt[crop], 1)
            center_arr[made] = cv2.flip(center[crop], 1)
            mask_arr[made] = cv2.flip((loss_mask[crop] * 255.0).astype(np.uint8), 1)
        else:
            x_arr[made] = stack[crop]
            target_arr[made] = tgt[crop]
            center_arr[made] = center[crop]
            mask_arr[made] = (loss_mask[crop] * 255.0).astype(np.uint8)
        made += 1
        if made == 1 or made % 100 == 0:
            print(f"cached={made}/{args.samples}", flush=True)
    out = args.cache_dir / f"masked_pool_p{args.patch}_n{args.samples}.npz"
    np.savez_compressed(out, x=x_arr, target=target_arr, center=center_arr, mask=mask_arr)
    meta = {"cache": str(out), "samples": args.samples, "patch": args.patch, "pool_root": str(root), "seed": args.seed}
    (args.cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


def to_tensor(torch, arr: np.ndarray):
    return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)


def laplacian(torch, value):
    import torch.nn.functional as F

    kernel = value.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    kernel = kernel.repeat(value.shape[1], 1, 1, 1)
    return F.conv2d(value, kernel, padding=1, groups=value.shape[1])


def build_model(torch, nn, F, width: int, residual_scale: float):
    class ResBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            groups = min(8, channels)
            self.net = nn.Sequential(
                nn.GroupNorm(groups, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(groups, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            )

        def forward(self, x):
            return x + self.net(x)

    class Refiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(16, width, 3, padding=1), ResBlock(width))
            self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
            self.enc2 = nn.Sequential(ResBlock(width * 2), ResBlock(width * 2))
            self.down2 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
            self.mid = nn.Sequential(ResBlock(width * 4), ResBlock(width * 4))
            self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 4, stride=2, padding=1)
            self.dec2 = nn.Sequential(ResBlock(width * 2))
            self.up1 = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
            self.dec1 = nn.Sequential(ResBlock(width))
            self.head = nn.Conv2d(width, 3, 3, padding=1)

        def forward(self, x, center, mask):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            mid = self.mid(self.down2(e2))
            d2 = self.dec2(self.up2(mid) + e2)
            d1 = self.dec1(self.up1(d2) + e1)
            raw = torch.tanh(self.head(d1))
            gate = (0.12 + 0.95 * mask).clamp(0.12, 1.0)
            residual = raw * residual_scale * gate
            return (center + residual).clamp(0.0, 1.0), residual

    return Refiner()


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    data_file = np.load(args.cache)
    x_data = data_file["x"]
    center_data = data_file["center"]
    target_data = data_file["target"]
    mask_data = data_file["mask"]
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = build_model(torch, nn, F, args.model_width, args.residual_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    rows = []
    n = x_data.shape[0]
    for step in range(1, args.steps + 1):
        ids = [rng.randrange(n) for _ in range(args.batch_size)]
        x = torch.stack([to_tensor(torch, x_data[i]) for i in ids]).to(device)
        center = torch.stack([to_tensor(torch, center_data[i]) for i in ids]).to(device)
        target = torch.stack([to_tensor(torch, target_data[i]) for i in ids]).to(device)
        mask = torch.stack([torch.from_numpy(mask_data[i].astype(np.float32) / 255.0)[None] for i in ids]).to(device)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            pred, residual = model(x, center, mask)
            weight = 1.0 + args.mask_weight * mask
            l1 = ((pred - target).abs() * weight).mean()
            edge = (laplacian(torch, pred) - laplacian(torch, target)).abs().mean()
            keep = ((pred - center).abs() * (1.0 - mask).clamp(0, 1)).mean()
            penalty = residual.abs().mean()
            loss = l1 + args.lambda_edge * edge + args.lambda_keep * keep + args.lambda_residual * penalty
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step == 1 or step % args.log_every == 0:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "l1": float(l1.detach().cpu()),
                "edge": float(edge.detach().cpu()),
                "keep": float(keep.detach().cpu()),
                "residual": float(penalty.detach().cpu()),
            }
            rows.append(row)
            print(" ".join(f"{k}={v:.6f}" if k != "step" else f"step={v}" for k, v in row.items()), flush=True)
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.out_dir / "masked_pool_refiner_last.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out_dir / "masked_pool_refiner_last.pt")
    if rows:
        with (args.out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def read_optional_folder(folder: Path | None, fallback: list[np.ndarray]) -> list[np.ndarray]:
    if folder is None:
        return fallback
    paths = sorted_frames(folder)
    if not paths:
        return fallback
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    if any(f is None for f in frames):
        return fallback
    return frames


def apply_folder(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    width = int(train_args.get("model_width", args.model_width))
    residual_scale = float(train_args.get("residual_scale", args.residual_scale))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch, nn, F, width, residual_scale).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    paths = sorted_frames(args.input_dir)
    if not paths:
        raise RuntimeError(f"no png frames in {args.input_dir}")
    medium = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    if any(f is None for f in medium):
        raise RuntimeError("failed to read input frames")
    long = read_optional_folder(args.long_dir, medium)
    actor = read_optional_folder(args.actor_dir, medium)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    strengths = [float(v) for v in args.strengths.split(",")]
    for strength in strengths:
        out_sub = args.out_dir / f"s{strength:.2f}".replace(".", "p")
        out_sub.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(paths):
            stack_np, center_np, infer = make_input_stack(medium, long, actor, i)
            x = to_tensor(torch, stack_np)[None].to(device)
            center = to_tensor(torch, center_np)[None].to(device)
            mask = torch.from_numpy(infer.astype(np.float32))[None, None].to(device)
            with torch.no_grad():
                pred, _ = model(x, center, mask)
            pred_np = (pred[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            mask3 = np.clip(infer[..., None] * strength, 0.0, 1.0)
            blended = np.clip(center_np.astype(np.float32) * (1.0 - mask3) + pred_np.astype(np.float32) * mask3, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_sub / path.name), blended)
        print(f"wrote {out_sub}", flush=True)


def audit_video(args: argparse.Namespace) -> None:
    base_paths = sorted_frames(args.base_dir)
    pred_dirs = [p for p in sorted(args.pred_root.iterdir()) if p.is_dir()]
    if not base_paths or not pred_dirs:
        raise RuntimeError("missing audit inputs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for pred_dir in pred_dirs:
        frames = []
        for p in base_paths:
            left = cv2.imread(str(p), cv2.IMREAD_COLOR)
            right = cv2.imread(str(pred_dir / p.name), cv2.IMREAD_COLOR)
            if left is None or right is None:
                continue
            if left.shape[:2] != right.shape[:2]:
                right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
            combo = np.concatenate([left, right], axis=1)
            cv2.putText(combo, "input", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(combo, pred_dir.name, (left.shape[1] + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
            frames.append(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB))
        out = args.out_dir / f"compare_{pred_dir.name}.mp4"
        imageio.mimsave(out, frames, fps=args.fps, quality=8)
        print(f"wrote {out}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--pool_root", type=Path, default=Path("/workspace/experiments/wan_ticft/bdd_candidate_router_train45_eval24/train_pool"))
    cache.add_argument("--cache_dir", type=Path, required=True)
    cache.add_argument("--samples", type=int, default=600)
    cache.add_argument("--patch", type=int, default=192)
    cache.add_argument("--seed", type=int, default=20260707)
    cache.add_argument("--clean", action="store_true")

    train_p = sub.add_parser("train")
    train_p.add_argument("--cache", type=Path, required=True)
    train_p.add_argument("--out_dir", type=Path, required=True)
    train_p.add_argument("--steps", type=int, default=400)
    train_p.add_argument("--batch_size", type=int, default=6)
    train_p.add_argument("--model_width", type=int, default=24)
    train_p.add_argument("--residual_scale", type=float, default=0.24)
    train_p.add_argument("--lr", type=float, default=2e-4)
    train_p.add_argument("--mask_weight", type=float, default=4.0)
    train_p.add_argument("--lambda_edge", type=float, default=0.12)
    train_p.add_argument("--lambda_keep", type=float, default=0.35)
    train_p.add_argument("--lambda_residual", type=float, default=0.03)
    train_p.add_argument("--log_every", type=int, default=50)
    train_p.add_argument("--seed", type=int, default=20260707)
    train_p.add_argument("--amp", action="store_true")
    train_p.add_argument("--clean", action="store_true")

    apply_p = sub.add_parser("apply")
    apply_p.add_argument("--checkpoint", type=Path, required=True)
    apply_p.add_argument("--input_dir", type=Path, required=True)
    apply_p.add_argument("--out_dir", type=Path, required=True)
    apply_p.add_argument("--long_dir", type=Path)
    apply_p.add_argument("--actor_dir", type=Path)
    apply_p.add_argument("--strengths", default="0.45,0.70,1.00")
    apply_p.add_argument("--model_width", type=int, default=24)
    apply_p.add_argument("--residual_scale", type=float, default=0.24)

    audit = sub.add_parser("audit")
    audit.add_argument("--base_dir", type=Path, required=True)
    audit.add_argument("--pred_root", type=Path, required=True)
    audit.add_argument("--out_dir", type=Path, required=True)
    audit.add_argument("--fps", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cv2.setNumThreads(2)
    if args.cmd == "cache":
        build_cache(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "apply":
        apply_folder(args)
    elif args.cmd == "audit":
        audit_video(args)


if __name__ == "__main__":
    main()
