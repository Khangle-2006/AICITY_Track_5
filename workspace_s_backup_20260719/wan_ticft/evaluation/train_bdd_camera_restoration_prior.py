"""Train/apply a BDD camera restoration prior for visual anti-blur.

The model learns from real BDD frames with synthetic blur/compression/noise
degradations. At inference it predicts only a high-frequency residual, so the
future-frame scaffold keeps its geometry while real-camera detail is restored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def numeric_pngs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"), key=lambda path: int(path.stem))


def read_bdd_videos(index_dir: Path, split: str, limit: int) -> list[Path]:
    path = index_dir / f"{split}_index.csv"
    videos: list[Path] = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("view_type") != "bdd":
                continue
            raw = row.get("video_path") or ""
            if not raw or raw in seen:
                continue
            video = Path(raw)
            if video.exists():
                videos.append(video)
                seen.add(raw)
            if limit and len(videos) >= limit:
                break
    return videos


def read_random_frame(video: Path, rng: random.Random) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count <= 0:
        cap.release()
        return None
    index = rng.randrange(0, count)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def crop_patch(image: np.ndarray, patch: int, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    if height < patch or width < patch:
        scale = max(patch / max(width, 1), patch / max(height, 1))
        image = cv2.resize(image, (math.ceil(width * scale), math.ceil(height * scale)), interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]
    top = rng.randrange(0, height - patch + 1)
    left = rng.randrange(0, width - patch + 1)
    return image[top : top + patch, left : left + patch]


def motion_kernel(size: int, angle: float) -> np.ndarray:
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    kernel /= max(kernel.sum(), 1e-6)
    return kernel


def degrade(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.copy()
    if rng.random() < 0.70:
        if rng.random() < 0.55:
            kernel = motion_kernel(rng.choice([5, 7, 9, 11]), rng.uniform(0, 180))
            out = cv2.filter2D(out, -1, kernel)
        else:
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=rng.uniform(0.8, 2.2))
    if rng.random() < 0.85:
        height, width = out.shape[:2]
        scale = rng.uniform(0.45, 0.85)
        small = cv2.resize(out, (max(8, int(width * scale)), max(8, int(height * scale))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.65:
        quality = rng.randint(45, 92)
        ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if rng.random() < 0.45:
        noise = rng.uniform(1.0, 5.0)
        out = np.clip(out.astype(np.float32) + rng.normalvariate(0, noise), 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        gamma = rng.uniform(0.86, 1.16)
        table = ((np.arange(256) / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
        out = cv2.LUT(out, table)
    return out


class RandomBDDFrameDataset:
    def __init__(self, videos: list[Path], patch: int, seed: int):
        self.videos = videos
        self.patch = patch
        self.rng = random.Random(seed)

    def sample(self):
        for _ in range(30):
            video = self.rng.choice(self.videos)
            frame = read_random_frame(video, self.rng)
            if frame is None:
                continue
            target = crop_patch(frame, self.patch, self.rng)
            source = degrade(target, self.rng)
            if self.rng.random() < 0.5:
                target = cv2.flip(target, 1)
                source = cv2.flip(source, 1)
            return source, target
        raise RuntimeError("could not sample a BDD frame")


def build_model(torch, nn, F, width: int, residual_scale: float):
    class ResBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.norm1 = nn.GroupNorm(8, channels)
            self.norm2 = nn.GroupNorm(8, channels)

        def forward(self, value):
            update = F.silu(self.norm1(self.conv1(value)))
            update = self.norm2(self.conv2(update))
            return F.silu(value + update)

    class Restorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.SiLU(), ResBlock(width))
            self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
            self.enc2 = nn.Sequential(nn.SiLU(), ResBlock(width * 2), ResBlock(width * 2))
            self.down2 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
            self.mid = nn.Sequential(nn.SiLU(), ResBlock(width * 4), ResBlock(width * 4), ResBlock(width * 4))
            self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 4, stride=2, padding=1)
            self.dec2 = nn.Sequential(nn.SiLU(), ResBlock(width * 2))
            self.up1 = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
            self.dec1 = nn.Sequential(nn.SiLU(), ResBlock(width))
            self.head = nn.Conv2d(width, 3, 3, padding=1)

        def forward(self, source):
            e1 = self.enc1(source)
            e2 = self.enc2(self.down1(e1))
            mid = self.mid(self.down2(e2))
            d2 = self.dec2(self.up2(mid) + e2)
            d1 = self.dec1(self.up1(d2) + e1)
            raw = torch.tanh(self.head(d1))
            low = F.avg_pool2d(raw, kernel_size=9, stride=1, padding=4)
            residual = (raw - low) * residual_scale
            return (source + residual).clamp(0.0, 1.0), residual

    return Restorer()


def to_tensor(torch, image: np.ndarray):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)


def from_tensor(tensor) -> np.ndarray:
    rgb = tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor((rgb * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr


def laplacian(torch, F, value):
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=value.device).view(1, 1, 3, 3)
    return F.conv2d(value, kernel.repeat(value.shape[1], 1, 1, 1), padding=1, groups=value.shape[1])


def train(args) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    videos = read_bdd_videos(args.index_dir, "train", args.video_limit)
    if not videos:
        raise RuntimeError("no BDD train videos found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    dataset = RandomBDDFrameDataset(videos, args.patch, args.seed)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model = build_model(torch, nn, F, args.model_width, args.residual_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for step in range(1, args.steps + 1):
        sources, targets = [], []
        for _ in range(args.batch_size):
            source, target = dataset.sample()
            sources.append(to_tensor(torch, source))
            targets.append(to_tensor(torch, target))
        source = torch.stack(sources).to(device)
        target = torch.stack(targets).to(device)
        pred, residual = model(source)
        l1 = (pred - target).abs().mean()
        edge = (laplacian(torch, F, pred) - laplacian(torch, F, target)).abs().mean()
        residual_penalty = residual.abs().mean()
        loss = l1 + args.lambda_edge * edge + args.lambda_residual * residual_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == 1:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "l1": float(l1.detach().cpu()),
                "edge": float(edge.detach().cpu()),
                "residual": float(residual_penalty.detach().cpu()),
            }
            history.append(row)
            print(" ".join(f"{key}={value:.6f}" if key != "step" else f"step={value}" for key, value in row.items()), flush=True)
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.out_dir / "restorer_last.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out_dir / "restorer_last.pt")
    with (args.out_dir / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def load_model(checkpoint: Path, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ckpt = torch.load(checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {})
    model = build_model(torch, nn, F, int(saved_args.get("model_width", 32)), float(saved_args.get("residual_scale", 0.18))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return torch, model


def enhance_image(torch, model, image: np.ndarray, device, tile_size: int, overlap: int, strength: float) -> np.ndarray:
    height, width = image.shape[:2]
    out = np.zeros((height, width, 3), np.float32)
    weight = np.zeros((height, width, 1), np.float32)
    step = max(1, tile_size - overlap)
    with torch.no_grad():
        for top in range(0, height, step):
            for left in range(0, width, step):
                bottom = min(height, top + tile_size)
                right = min(width, left + tile_size)
                top0 = max(0, bottom - tile_size)
                left0 = max(0, right - tile_size)
                tile = image[top0:bottom, left0:right]
                tensor = to_tensor(torch, tile).unsqueeze(0).to(device)
                pred, _ = model(tensor)
                restored_u8 = from_tensor(pred[0])
                if strength != 1.0:
                    restored_u8 = np.clip(tile.astype(np.float32) + strength * (restored_u8.astype(np.float32) - tile.astype(np.float32)), 0, 255).astype(np.uint8)
                restored = restored_u8.astype(np.float32)
                mask = np.ones((restored.shape[0], restored.shape[1], 1), np.float32)
                if overlap > 0:
                    ramp_y = np.minimum(np.linspace(0, 1, restored.shape[0])[:, None], np.linspace(1, 0, restored.shape[0])[:, None])
                    ramp_x = np.minimum(np.linspace(0, 1, restored.shape[1])[None, :], np.linspace(1, 0, restored.shape[1])[None, :])
                    mask = np.maximum(0.15, np.minimum(ramp_y, ramp_x))[..., None].astype(np.float32)
                out[top0:bottom, left0:right] += restored * mask
                weight[top0:bottom, left0:right] += mask
    return np.clip(out / np.maximum(weight, 1e-6), 0, 255).astype(np.uint8)


def apply(args) -> None:
    import torch

    device = torch.device("cuda")
    torch_mod, model = load_model(args.checkpoint, device)
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = [folder for folder in sorted(args.input_dir.iterdir()) if folder.is_dir()]
    if args.sample_limit:
        samples = samples[: args.sample_limit]
    rows = []
    for sample in samples:
        out_sample = args.out_dir / sample.name
        out_sample.mkdir(parents=True, exist_ok=True)
        frames = numeric_pngs(sample)
        for frame_path in frames:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            restored = enhance_image(torch_mod, model, image, device, args.tile_size, args.overlap, args.strength)
            cv2.imwrite(str(out_sample / frame_path.name), restored)
        rows.append({"output_name": sample.name, "frames_found": len(frames), "status": "ok"})
        print(f"apply {sample.name} frames={len(frames)}", flush=True)
    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["output_name", "frames_found", "status"])
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--index_dir", type=Path, default=Path("/workspace/experiments/index"))
    train_parser.add_argument("--out_dir", type=Path, required=True)
    train_parser.add_argument("--video_limit", type=int, default=800)
    train_parser.add_argument("--patch", type=int, default=192)
    train_parser.add_argument("--batch_size", type=int, default=12)
    train_parser.add_argument("--steps", type=int, default=1600)
    train_parser.add_argument("--model_width", type=int, default=32)
    train_parser.add_argument("--residual_scale", type=float, default=0.18)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--lambda_edge", type=float, default=0.18)
    train_parser.add_argument("--lambda_residual", type=float, default=0.03)
    train_parser.add_argument("--log_every", type=int, default=50)
    train_parser.add_argument("--seed", type=int, default=20260705)

    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("--checkpoint", type=Path, required=True)
    apply_parser.add_argument("--input_dir", type=Path, required=True)
    apply_parser.add_argument("--out_dir", type=Path, required=True)
    apply_parser.add_argument("--sample_limit", type=int, default=0)
    apply_parser.add_argument("--tile_size", type=int, default=384)
    apply_parser.add_argument("--overlap", type=int, default=64)
    apply_parser.add_argument("--strength", type=float, default=1.0)
    apply_parser.add_argument("--clean", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "apply":
        apply(args)


if __name__ == "__main__":
    main()
