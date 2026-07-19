"""Fast temporal deghost refiner for future-frame artifacts.

The model is trained on cached BDD clips with synthetic temporal double
exposure. It receives a short corrupted frame window and predicts a residual
for the center frame. This targets the failure seen in generated Track 5
videos: an object position is blended across time, then the stale position
turns into a dragged dark trail.
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
import imageio.v2 as imageio
import numpy as np


def read_bdd_videos(index_dir: Path, split: str, limit: int) -> list[Path]:
    videos: list[Path] = []
    seen: set[str] = set()
    with (index_dir / f"{split}_index.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("view_type") != "bdd":
                continue
            raw = row.get("video_path") or ""
            if raw in seen:
                continue
            path = Path(raw)
            if path.exists():
                videos.append(path)
                seen.add(raw)
            if limit and len(videos) >= limit:
                break
    return videos


def sample_clip_patch(video: Path, frames: int, patch: int, rng: random.Random) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count < frames + 2:
        cap.release()
        return None
    start = rng.randrange(0, count - frames)
    raw_frames = []
    for idx in range(start, start + frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None
        raw_frames.append(frame)
    cap.release()
    h, w = raw_frames[0].shape[:2]
    if h < patch or w < patch:
        scale = max(patch / h, patch / w)
        new_w, new_h = int(math.ceil(w * scale)), int(math.ceil(h * scale))
        raw_frames = [cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for frame in raw_frames]
        h, w = raw_frames[0].shape[:2]
    x = rng.randrange(0, w - patch + 1)
    y = rng.randrange(0, h - patch + 1)
    clip = [frame[y : y + patch, x : x + patch] for frame in raw_frames]
    if rng.random() < 0.5:
        clip = [cv2.flip(frame, 1) for frame in clip]
    return np.stack(clip, axis=0)


def build_cache(args) -> None:
    rng = random.Random(args.seed)
    videos = read_bdd_videos(args.index_dir, args.split, args.video_limit)
    if not videos:
        raise RuntimeError("no BDD videos found")
    if args.clean and args.cache_dir.exists():
        shutil.rmtree(args.cache_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.cache_dir / f"clips_t{args.frames}_p{args.patch}_n{args.samples}.npy"
    meta_path = args.cache_dir / "cache_meta.json"
    clips = np.empty((args.samples, args.frames, args.patch, args.patch, 3), dtype=np.uint8)
    made = 0
    tries = 0
    while made < args.samples and tries < args.samples * 40:
        tries += 1
        clip = sample_clip_patch(rng.choice(videos), args.frames, args.patch, rng)
        if clip is None:
            continue
        clips[made] = clip
        made += 1
        if made == 1 or made % 100 == 0:
            print(f"cached={made}/{args.samples}", flush=True)
    if made < args.samples:
        clips = clips[:made]
    np.save(out_path, clips)
    meta_path.write_text(
        json.dumps(
            {
                "path": str(out_path),
                "samples": int(clips.shape[0]),
                "frames": args.frames,
                "patch": args.patch,
                "split": args.split,
                "video_limit": args.video_limit,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path} shape={clips.shape}", flush=True)


def temporal_ghost_degrade(clip: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Return corrupted clip and center-frame ghost mask."""
    frames, h, w, _ = clip.shape
    source = clip.astype(np.float32).copy()
    center = frames // 2
    center_mask = np.zeros((h, w), np.float32)
    direction = rng.choice([-1, 1])
    lag = rng.choice([1, 1, 2])
    for t in range(frames):
        stale_t = max(0, t - lag) if rng.random() < 0.75 else min(frames - 1, t + lag)
        stale = clip[stale_t]
        dx = direction * rng.uniform(2.0, 0.13 * w)
        dy = rng.uniform(-0.035 * h, 0.045 * h)
        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        stale = cv2.warpAffine(stale, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        if rng.random() < 0.8:
            stale = cv2.GaussianBlur(stale, (0, 0), sigmaX=rng.uniform(0.4, 1.8))
        diff = cv2.cvtColor(cv2.absdiff(stale, clip[t]), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mask = np.clip((diff - rng.uniform(0.015, 0.045)) / rng.uniform(0.08, 0.16), 0.0, 1.0)
        k = rng.choice([5, 7, 9])
        mask = cv2.GaussianBlur(mask, (k, k), sigmaX=rng.uniform(0.8, 2.2))
        alpha = rng.uniform(0.12, 0.36)
        if rng.random() < 0.45:
            stale = (stale.astype(np.float32) * rng.uniform(0.68, 0.92)).astype(np.float32)
        else:
            stale = stale.astype(np.float32)
        m3 = mask[..., None] * alpha
        source[t] = source[t] * (1.0 - m3) + stale * m3
        if t == center:
            center_mask = np.maximum(center_mask, mask)
    if rng.random() < 0.45:
        # Add mild generated-video softness without making the target geometry change.
        for t in range(frames):
            small_scale = rng.uniform(0.72, 0.92)
            small = cv2.resize(
                source[t].astype(np.uint8),
                (max(8, int(w * small_scale)), max(8, int(h * small_scale))),
                interpolation=cv2.INTER_AREA,
            )
            source[t] = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    if rng.random() < 0.35:
        for t in range(frames):
            quality = rng.randint(62, 96)
            ok, enc = cv2.imencode(".jpg", source[t].astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                source[t] = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)
    return np.clip(source, 0, 255).astype(np.uint8), center_mask.astype(np.float32)


def to_tensor(torch, arr: np.ndarray):
    return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)


def laplacian(torch, value):
    import torch.nn.functional as F

    kernel = value.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    kernel = kernel.repeat(value.shape[1], 1, 1, 1)
    return F.conv2d(value, kernel, padding=1, groups=value.shape[1])


def build_temporal_model(torch, nn, F, frames: int, width: int, residual_scale: float, residual_mode: str = "highpass"):
    class ResBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            groups = min(8, channels)
            self.norm1 = nn.GroupNorm(groups, channels)
            self.norm2 = nn.GroupNorm(groups, channels)

        def forward(self, value):
            update = F.silu(self.norm1(self.conv1(value)))
            update = self.norm2(self.conv2(update))
            return F.silu(value + update)

    class TemporalRestorer(nn.Module):
        def __init__(self):
            super().__init__()
            in_ch = frames * 3 + 1
            self.enc1 = nn.Sequential(nn.Conv2d(in_ch, width, 3, padding=1), nn.SiLU(), ResBlock(width))
            self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
            self.enc2 = nn.Sequential(nn.SiLU(), ResBlock(width * 2), ResBlock(width * 2))
            self.down2 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
            self.mid = nn.Sequential(nn.SiLU(), ResBlock(width * 4), ResBlock(width * 4), ResBlock(width * 4))
            self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 4, stride=2, padding=1)
            self.dec2 = nn.Sequential(nn.SiLU(), ResBlock(width * 2))
            self.up1 = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
            self.dec1 = nn.Sequential(nn.SiLU(), ResBlock(width))
            self.head = nn.Conv2d(width, 3, 3, padding=1)

        def forward(self, stacked, center, mask=None):
            e1 = self.enc1(stacked)
            e2 = self.enc2(self.down1(e1))
            mid = self.mid(self.down2(e2))
            d2 = self.dec2(self.up2(mid) + e2)
            d1 = self.dec1(self.up1(d2) + e1)
            raw = torch.tanh(self.head(d1))
            if residual_mode == "full":
                if mask is None:
                    gate = 1.0
                else:
                    gate = (0.20 + 0.95 * mask).clamp(0.20, 1.15)
                residual = raw * residual_scale * gate
            else:
                low = F.avg_pool2d(raw, kernel_size=9, stride=1, padding=4)
                residual = (raw - low) * residual_scale
            return (center + residual).clamp(0.0, 1.0), residual

    return TemporalRestorer()


def train(args) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cache = np.load(args.cache, mmap_mode="r")
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = build_temporal_model(torch, nn, F, cache.shape[1], args.model_width, args.residual_scale, args.residual_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    history = []
    center_idx = cache.shape[1] // 2
    for step in range(1, args.steps + 1):
        stacked_batch, center_batch, target_batch, mask_batch = [], [], [], []
        for _ in range(args.batch_size):
            idx = rng.randrange(0, cache.shape[0])
            clean_clip = np.array(cache[idx])
            source_clip, mask = temporal_ghost_degrade(clean_clip, rng)
            target = clean_clip[center_idx]
            center = source_clip[center_idx]
            stacked = np.concatenate([source_clip[t] for t in range(source_clip.shape[0])], axis=2)
            stacked_batch.append(to_tensor(torch, stacked))
            center_batch.append(to_tensor(torch, center))
            target_batch.append(to_tensor(torch, target))
            mask_batch.append(torch.from_numpy(mask[None]))
        stacked = torch.stack(stacked_batch).to(device)
        center = torch.stack(center_batch).to(device)
        target = torch.stack(target_batch).to(device)
        mask = torch.stack(mask_batch).to(device)
        stacked = torch.cat([stacked, mask], dim=1)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            pred, residual = model(stacked, center, mask)
            weight = 1.0 + args.mask_weight * mask
            l1 = ((pred - target).abs() * weight).mean()
            edge = (laplacian(torch, pred) - laplacian(torch, target)).abs().mean()
            center_keep = ((pred - center).abs() * (1.0 - mask).clamp(0, 1)).mean()
            residual_penalty = residual.abs().mean()
            loss = l1 + args.lambda_edge * edge + args.lambda_keep * center_keep + args.lambda_residual * residual_penalty
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % args.log_every == 0:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "l1": float(l1.detach().cpu()),
                "edge": float(edge.detach().cpu()),
                "keep": float(center_keep.detach().cpu()),
                "residual": float(residual_penalty.detach().cpu()),
            }
            history.append(row)
            print(" ".join(f"{k}={v:.6f}" if k != "step" else f"step={v}" for k, v in row.items()), flush=True)
            torch.save({"model": model.state_dict(), "args": vars(args), "frames": int(cache.shape[1])}, args.out_dir / "temporal_deghost_last.pt")
    torch.save({"model": model.state_dict(), "args": vars(args), "frames": int(cache.shape[1])}, args.out_dir / "temporal_deghost_last.pt")
    if history:
        with (args.out_dir / "history.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)


def sorted_pngs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)


def infer_sequence(args) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    frames = int(ckpt.get("frames") or ckpt.get("args", {}).get("frames") or 5)
    width = int(ckpt["args"].get("model_width", args.model_width))
    residual_scale = float(ckpt["args"].get("residual_scale", args.residual_scale))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual_mode = str(ckpt["args"].get("residual_mode", "highpass"))
    model = build_temporal_model(torch, nn, F, frames, width, residual_scale, residual_mode).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    paths = sorted_pngs(args.input_dir)
    if not paths:
        raise RuntimeError(f"no png frames in {args.input_dir}")
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if any(image is None for image in images):
        raise RuntimeError("failed reading one or more frames")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    half = frames // 2
    strengths = [float(v) for v in args.strengths.split(",")]
    for strength in strengths:
        out_sub = args.out_dir / f"s{strength:.2f}".replace(".", "p")
        out_sub.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            window = []
            for off in range(-half, half + 1):
                j = min(len(images) - 1, max(0, i + off))
                window.append(images[j])
            stacked_np = np.concatenate(window, axis=2)
            center_np = image
            stacked = to_tensor(torch, stacked_np)[None].to(device)
            center = to_tensor(torch, center_np)[None].to(device)
            gray_prev = cv2.cvtColor(window[max(0, half - 1)], cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_curr = cv2.cvtColor(center_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_next = cv2.cvtColor(window[min(frames - 1, half + 1)], cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask_np = np.maximum(np.abs(gray_curr - gray_prev), np.abs(gray_next - gray_curr)) / 255.0
            mask_np = cv2.GaussianBlur(np.clip((mask_np - 0.015) / 0.18, 0, 1), (9, 9), 1.5).astype(np.float32)
            mask = torch.from_numpy(mask_np[None, None]).to(device)
            with torch.no_grad():
                pred, _ = model(torch.cat([stacked, mask], dim=1), center, mask)
            pred_np = (pred[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            blended = np.clip(center_np.astype(np.float32) * (1.0 - strength) + pred_np.astype(np.float32) * strength, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_sub / paths[i].name), blended)
        print(f"wrote {out_sub}", flush=True)


def make_audit(args) -> None:
    input_paths = sorted_pngs(args.input_dir)
    output_dirs = [p for p in sorted(args.pred_dir.iterdir()) if p.is_dir()]
    if not input_paths or not output_dirs:
        raise RuntimeError("missing audit inputs")
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = [int(v) for v in args.frames_to_show.split(",")]
    labels = ["input"] + [p.name for p in output_dirs]
    rows = []
    for frame_id in frame_ids:
        cells = []
        sources = [args.input_dir] + output_dirs
        for label, folder in zip(labels, sources):
            path = folder / f"{frame_id}.png"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                image = np.zeros_like(cv2.imread(str(input_paths[0]), cv2.IMREAD_COLOR))
            cv2.putText(image, f"{label} f{frame_id}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
            cells.append(image)
        rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(args.audit_dir / "temporal_deghost_keyframes.jpg"), sheet)
    for out_dir in output_dirs:
        writer_path = args.audit_dir / f"compare_{out_dir.name}.mp4"
        frames = []
        for path in input_paths:
            left = cv2.imread(str(path), cv2.IMREAD_COLOR)
            right = cv2.imread(str(out_dir / path.name), cv2.IMREAD_COLOR)
            if left is None or right is None:
                continue
            combo = np.concatenate([left, right], axis=1)
            cv2.putText(combo, "input", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(combo, out_dir.name, (left.shape[1] + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
            frames.append(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB))
        imageio.mimsave(writer_path, frames, fps=args.fps, quality=8)
        print(f"wrote {writer_path}", flush=True)
    print(f"wrote {args.audit_dir / 'temporal_deghost_keyframes.jpg'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cache = sub.add_parser("cache")
    cache.add_argument("--index_dir", type=Path, default=Path("/workspace/experiments/index"))
    cache.add_argument("--split", default="train")
    cache.add_argument("--cache_dir", type=Path, required=True)
    cache.add_argument("--video_limit", type=int, default=800)
    cache.add_argument("--samples", type=int, default=1200)
    cache.add_argument("--frames", type=int, default=5)
    cache.add_argument("--patch", type=int, default=192)
    cache.add_argument("--seed", type=int, default=20260706)
    cache.add_argument("--clean", action="store_true")

    train_p = sub.add_parser("train")
    train_p.add_argument("--cache", type=Path, required=True)
    train_p.add_argument("--out_dir", type=Path, required=True)
    train_p.add_argument("--steps", type=int, default=1000)
    train_p.add_argument("--batch_size", type=int, default=8)
    train_p.add_argument("--model_width", type=int, default=32)
    train_p.add_argument("--residual_scale", type=float, default=0.30)
    train_p.add_argument("--residual_mode", choices=["highpass", "full"], default="highpass")
    train_p.add_argument("--lr", type=float, default=2e-4)
    train_p.add_argument("--mask_weight", type=float, default=2.5)
    train_p.add_argument("--lambda_edge", type=float, default=0.18)
    train_p.add_argument("--lambda_keep", type=float, default=0.18)
    train_p.add_argument("--lambda_residual", type=float, default=0.025)
    train_p.add_argument("--log_every", type=int, default=50)
    train_p.add_argument("--seed", type=int, default=20260706)
    train_p.add_argument("--amp", action="store_true")
    train_p.add_argument("--clean", action="store_true")

    apply_p = sub.add_parser("apply")
    apply_p.add_argument("--checkpoint", type=Path, required=True)
    apply_p.add_argument("--input_dir", type=Path, required=True)
    apply_p.add_argument("--out_dir", type=Path, required=True)
    apply_p.add_argument("--strengths", default="0.6,1.0,1.35")
    apply_p.add_argument("--model_width", type=int, default=32)
    apply_p.add_argument("--residual_scale", type=float, default=0.30)

    audit = sub.add_parser("audit")
    audit.add_argument("--input_dir", type=Path, required=True)
    audit.add_argument("--pred_dir", type=Path, required=True)
    audit.add_argument("--audit_dir", type=Path, required=True)
    audit.add_argument("--frames_to_show", default="5,6,37")
    audit.add_argument("--fps", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "cache":
        build_cache(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "apply":
        infer_sequence(args)
    elif args.cmd == "audit":
        make_audit(args)


if __name__ == "__main__":
    main()
