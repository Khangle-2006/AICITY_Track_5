from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_targets(cv2, row: dict[str, Any]):
    return [img for path in (row.get("frames") or {}).get("target", []) if (img := read_rgb(cv2, Path(path))) is not None]


def load_last_condition(cv2, row: dict[str, Any]):
    paths = (row.get("frames") or {}).get("condition", [])
    return read_rgb(cv2, Path(paths[-1])) if paths else None


def load_base_frames(cv2, row: dict[str, Any], video: Path):
    offsets = map_target_offsets(row, video_fps(cv2, video))
    frame_map = read_video_rgb_frames(cv2, video, offsets)
    return [frame_map[index] for index in offsets if index in frame_map]


def resize(cv2, img, width: int, height: int):
    if img.shape[1] == width and img.shape[0] == height:
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def make_tensor(torch, np, base, condition, frame_index: int, frame_count: int):
    t = 0.0 if frame_count <= 1 else frame_index / float(frame_count - 1)
    time = np.full((base.shape[0], base.shape[1], 1), t * 255.0, dtype=np.float32)
    x = np.concatenate([base.astype(np.float32), condition.astype(np.float32), time], axis=2)
    return torch.from_numpy(x / 255.0).permute(2, 0, 1)


def image_tensor(torch, np, img):
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def tensor_to_rgb(np, tensor):
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def build_model(nn, in_channels: int, width: int, residual_scale: float):
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.residual_scale = residual_scale
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, width, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, 3, 3, padding=1),
                nn.Tanh(),
            )

        def forward(self, x):
            base = x[:, :3]
            return (base + self.net(x) * self.residual_scale).clamp(0, 1)

    return Adapter()


def ssim_loss(torch, F, pred, target):
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(pred, 7, stride=1, padding=3)
    mu_y = F.avg_pool2d(target, 7, stride=1, padding=3)
    sigma_x = F.avg_pool2d(pred * pred, 7, stride=1, padding=3) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 7, stride=1, padding=3) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 7, stride=1, padding=3) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return (1.0 - ssim.clamp(0, 1)).mean()


def gradient_loss(torch, pred, target):
    return (
        (pred[:, :, :, 1:] - pred[:, :, :, :-1] - target[:, :, :, 1:] + target[:, :, :, :-1]).abs().mean()
        + (pred[:, :, 1:, :] - pred[:, :, :-1, :] - target[:, :, 1:, :] + target[:, :, :-1, :]).abs().mean()
    )


def make_items(cv2, np, torch, rows, report, width: int, height: int):
    items = []
    for row in rows:
        sample_id = row["sample_id"]
        video = report.get(sample_id)
        if video is None:
            continue
        targets = load_targets(cv2, row)
        bases = load_base_frames(cv2, row, video)
        condition = load_last_condition(cv2, row)
        if condition is None or len(targets) != len(bases):
            continue
        condition = resize(cv2, condition, width, height)
        for index, (base, target) in enumerate(zip(bases, targets)):
            base = resize(cv2, base, width, height)
            target = resize(cv2, target, width, height)
            items.append(
                {
                    "x": make_tensor(torch, np, base, condition, index, len(bases)),
                    "y": image_tensor(torch, np, target),
                    "base": image_tensor(torch, np, base),
                }
            )
    return items


def train_adapter(torch, F, model, items, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(items)
        model.train()
        totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "grad": 0.0, "res": 0.0}
        count = 0
        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            x = torch.stack([item["x"] for item in batch]).to(device)
            y = torch.stack([item["y"] for item in batch]).to(device)
            b = torch.stack([item["base"] for item in batch]).to(device)
            pred = model(x)
            l1 = (pred - y).abs().mean()
            sl = ssim_loss(torch, F, pred, y)
            gl = gradient_loss(torch, pred, y)
            rl = (pred - b).abs().mean()
            loss = l1 + args.lambda_ssim * sl + args.lambda_grad * gl + args.lambda_residual * rl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = len(batch)
            totals["loss"] += float(loss.detach().cpu()) * bs
            totals["l1"] += float(l1.detach().cpu()) * bs
            totals["ssim"] += float(sl.detach().cpu()) * bs
            totals["grad"] += float(gl.detach().cpu()) * bs
            totals["res"] += float(rl.detach().cpu()) * bs
            count += bs
        row = {"epoch": epoch, **{k: v / max(1, count) for k, v in totals.items()}}
        history.append(row)
        print(
            f"epoch={epoch} loss={row['loss']:.6f} l1={row['l1']:.6f} "
            f"ssim={row['ssim']:.6f} grad={row['grad']:.6f} res={row['res']:.6f}",
            flush=True,
        )
    return history


def train_adapter_feature(torch, F, model, train_sequences, args, device):
    from torchvision.models import Inception_V3_Weights, inception_v3
    from torchvision.models.video import R3D_18_Weights, r3d_18

    image_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    image_model.fc = torch.nn.Identity()
    image_model.eval().to(device)
    for p in image_model.parameters():
        p.requires_grad_(False)

    video_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    video_model.fc = torch.nn.Identity()
    video_model.eval().to(device)
    for p in video_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    history = []

    image_mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    image_std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    video_mean = torch.tensor((0.43216, 0.394666, 0.37645), device=device)[None, :, None, None, None]
    video_std = torch.tensor((0.22803, 0.22145, 0.216989), device=device)[None, :, None, None, None]

    def image_features(x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return image_model((x - image_mean) / image_std)

    def video_features(x):
        # x: B,T,C,H,W -> B,C,T,H,W
        x = x.permute(0, 2, 1, 3, 4)
        b, c, t, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        x = x.reshape(b, t, c, 112, 112).permute(0, 2, 1, 3, 4)
        return video_model((x - video_mean) / video_std)

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_sequences)
        model.train()
        totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "img": 0.0, "vid": 0.0, "temp": 0.0, "res": 0.0}
        count = 0
        for seq in train_sequences:
            if len(seq) < args.clip_frames:
                continue
            max_start = len(seq) - args.clip_frames
            start = rng.randint(0, max_start)
            clip = seq[start : start + args.clip_frames]
            x = torch.stack([item["x"] for item in clip]).to(device)
            y = torch.stack([item["y"] for item in clip]).to(device)
            b = torch.stack([item["base"] for item in clip]).to(device)
            pred = model(x)
            l1 = (pred - y).abs().mean()
            sl = ssim_loss(torch, F, pred, y)
            rl = (pred - b).abs().mean()
            with torch.no_grad():
                y_img = image_features(y)
                y_vid = video_features(y.unsqueeze(0))
            img = (image_features(pred) - y_img).abs().mean()
            vid = (video_features(pred.unsqueeze(0)) - y_vid).abs().mean()
            temp = ((pred[1:] - pred[:-1]) - (y[1:] - y[:-1])).abs().mean()
            loss = (
                args.lambda_l1 * l1
                + args.lambda_ssim * sl
                + args.lambda_image_feature * img
                + args.lambda_video_feature * vid
                + args.lambda_temporal * temp
                + args.lambda_residual * rl
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["l1"] += float(l1.detach().cpu())
            totals["ssim"] += float(sl.detach().cpu())
            totals["img"] += float(img.detach().cpu())
            totals["vid"] += float(vid.detach().cpu())
            totals["temp"] += float(temp.detach().cpu())
            totals["res"] += float(rl.detach().cpu())
            count += 1
            if args.max_steps and count >= args.max_steps:
                break
        row = {"epoch": epoch, **{k: v / max(1, count) for k, v in totals.items()}}
        history.append(row)
        print(
            f"epoch={epoch} loss={row['loss']:.6f} l1={row['l1']:.6f} ssim={row['ssim']:.6f} "
            f"img={row['img']:.6f} vid={row['vid']:.6f} temp={row['temp']:.6f} res={row['res']:.6f}",
            flush=True,
        )
    return history


def train_adapter_distribution(torch, F, model, train_sequences, args, device):
    from torchvision.models import Inception_V3_Weights, inception_v3
    from torchvision.models.video import R3D_18_Weights, r3d_18

    image_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    image_model.fc = torch.nn.Identity()
    image_model.eval().to(device)
    for p in image_model.parameters():
        p.requires_grad_(False)

    video_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    video_model.fc = torch.nn.Identity()
    video_model.eval().to(device)
    for p in video_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    history = []

    image_mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    image_std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    video_mean = torch.tensor((0.43216, 0.394666, 0.37645), device=device)[None, :, None, None, None]
    video_std = torch.tensor((0.22803, 0.22145, 0.216989), device=device)[None, :, None, None, None]

    def image_features(x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return image_model((x - image_mean) / image_std)

    def video_features(x):
        # x: B,T,C,H,W -> B,C,T,H,W
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w)
        flat = F.interpolate(flat, size=(112, 112), mode="bilinear", align_corners=False)
        x = flat.reshape(b, t, c, 112, 112).permute(0, 2, 1, 3, 4)
        return video_model((x - video_mean) / video_std)

    def stat_loss(pred_feat, target_feat):
        pred = pred_feat.float()
        target = target_feat.float()
        mean = (pred.mean(dim=0) - target.mean(dim=0)).pow(2).mean()
        var = (pred.var(dim=0, unbiased=False) - target.var(dim=0, unbiased=False)).pow(2).mean()
        return mean + var

    valid_sequences = [seq for seq in train_sequences if len(seq) >= args.clip_frames]
    if not valid_sequences:
        raise RuntimeError("no training sequences with enough frames")

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "img": 0.0, "vid": 0.0, "temp": 0.0, "dist": 0.0, "res": 0.0}
        count = 0
        steps = args.max_steps or max(1, len(valid_sequences) * 2)
        for _step in range(steps):
            batch_sequences = [rng.choice(valid_sequences) for _ in range(args.sequence_batch)]
            xs = []
            ys = []
            bs = []
            for seq in batch_sequences:
                start = rng.randint(0, len(seq) - args.clip_frames)
                clip = seq[start : start + args.clip_frames]
                xs.append(torch.stack([item["x"] for item in clip]))
                ys.append(torch.stack([item["y"] for item in clip]))
                bs.append(torch.stack([item["base"] for item in clip]))
            x = torch.stack(xs).to(device)  # B,T,C,H,W
            y = torch.stack(ys).to(device)
            b = torch.stack(bs).to(device)
            batch, frames, channels, height, width = x.shape
            pred = model(x.reshape(batch * frames, channels, height, width)).reshape(batch, frames, 3, height, width)

            pred_flat = pred.reshape(batch * frames, 3, height, width)
            y_flat = y.reshape(batch * frames, 3, height, width)
            b_flat = b.reshape(batch * frames, 3, height, width)
            l1 = (pred_flat - y_flat).abs().mean()
            sl = ssim_loss(torch, F, pred_flat, y_flat)
            rl = (pred_flat - b_flat).abs().mean()
            temp = ((pred[:, 1:] - pred[:, :-1]) - (y[:, 1:] - y[:, :-1])).abs().mean()

            with torch.no_grad():
                y_img = image_features(y_flat)
                y_vid = video_features(y)
            pred_img = image_features(pred_flat)
            pred_vid = video_features(pred)
            img = stat_loss(pred_img, y_img)
            vid = stat_loss(pred_vid, y_vid)
            dist = img + vid

            loss = (
                args.lambda_l1 * l1
                + args.lambda_ssim * sl
                + args.lambda_image_feature * img
                + args.lambda_video_feature * vid
                + args.lambda_temporal * temp
                + args.lambda_distribution * dist
                + args.lambda_residual * rl
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["l1"] += float(l1.detach().cpu())
            totals["ssim"] += float(sl.detach().cpu())
            totals["img"] += float(img.detach().cpu())
            totals["vid"] += float(vid.detach().cpu())
            totals["temp"] += float(temp.detach().cpu())
            totals["dist"] += float(dist.detach().cpu())
            totals["res"] += float(rl.detach().cpu())
            count += 1
        row = {"epoch": epoch, **{k: v / max(1, count) for k, v in totals.items()}}
        history.append(row)
        print(
            f"epoch={epoch} loss={row['loss']:.6f} l1={row['l1']:.6f} ssim={row['ssim']:.6f} "
            f"img={row['img']:.6f} vid={row['vid']:.6f} dist={row['dist']:.6f} "
            f"temp={row['temp']:.6f} res={row['res']:.6f}",
            flush=True,
        )
    return history


def write_history(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "loss", "l1", "ssim", "grad", "img", "vid", "temp", "dist", "res"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def make_sequences(cv2, np, torch, rows, report, width: int, height: int):
    sequences = []
    for row in rows:
        sample_id = row["sample_id"]
        video = report.get(sample_id)
        if video is None:
            continue
        targets = load_targets(cv2, row)
        bases = load_base_frames(cv2, row, video)
        condition = load_last_condition(cv2, row)
        if condition is None or len(targets) != len(bases):
            continue
        condition = resize(cv2, condition, width, height)
        seq = []
        for index, (base, target) in enumerate(zip(bases, targets)):
            base = resize(cv2, base, width, height)
            target = resize(cv2, target, width, height)
            seq.append(
                {
                    "x": make_tensor(torch, np, base, condition, index, len(bases)),
                    "y": image_tensor(torch, np, target),
                    "base": image_tensor(torch, np, base),
                }
            )
        sequences.append(seq)
    return sequences


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # Some OpenCV/MP4 combinations expose one fewer readable frame after muxing.
    # A duplicate tail frame keeps timestamp-based validation from dropping the
    # final target while preserving all intended frame indices.
    writer.write(cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    writer.release()


def infer_rows(cv2, np, torch, model, rows, report, out_dir: Path, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    report_rows = []
    model.eval()
    with torch.no_grad():
        for pos, row in enumerate(rows, start=1):
            sample_id = row["sample_id"]
            video = report.get(sample_id)
            if video is None:
                continue
            bases = load_base_frames(cv2, row, video)
            condition = load_last_condition(cv2, row)
            if condition is None or not bases:
                continue
            fps = video_fps(cv2, video) or 30.0
            frames = []
            for index, base in enumerate(bases):
                cond = condition
                if cond.shape[:2] != base.shape[:2]:
                    cond = cv2.resize(cond, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)
                x = make_tensor(torch, np, base, cond, index, len(bases)).unsqueeze(0).to(device)
                pred = model(x)[0]
                frames.append(tensor_to_rgb(np, pred))
            out_path = out_dir / "videos" / f"{sample_id}.mp4"
            write_video(cv2, frames, out_path, fps)
            report_rows.append({"sample_id": sample_id, "video": str(out_path), "status": "ok", "future_frames": len(frames)})
            print(f"infer [{pos}/{len(rows)}] {sample_id} | ok", flush=True)
    with (out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "future_frames"])
        writer.writeheader()
        writer.writerows(report_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a metric-loss residual adapter on generated validation videos.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--train_rows", type=int, default=16)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--model_width", type=int, default=48)
    parser.add_argument("--residual_scale", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_ssim", type=float, default=0.35)
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_grad", type=float, default=0.08)
    parser.add_argument("--lambda_residual", type=float, default=0.10)
    parser.add_argument("--lambda_image_feature", type=float, default=0.0)
    parser.add_argument("--lambda_video_feature", type=float, default=0.0)
    parser.add_argument("--lambda_temporal", type=float, default=0.0)
    parser.add_argument("--lambda_distribution", type=float, default=0.0)
    parser.add_argument("--clip_frames", type=int, default=8)
    parser.add_argument("--sequence_batch", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--feature_loss", action="store_true")
    parser.add_argument("--distribution_loss", action="store_true")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    torch, nn, F = import_torch()
    cv2, np = import_cv2_np()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"][: args.limit]
    train_rows = rows[: args.train_rows]
    eval_rows = rows[args.train_rows :]
    if not train_rows or not eval_rows:
        raise RuntimeError("need non-empty train and eval split")

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train_manifest.jsonl", train_rows)
    write_jsonl(args.out_dir / "eval_manifest.jsonl", eval_rows)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.out_dir / "args.json").write_text(json.dumps(serializable_args, indent=2), encoding="utf-8")

    report = read_report(args.base_report)
    model = build_model(nn, 7, args.model_width, args.residual_scale).to(device)
    if args.distribution_loss:
        sequences = make_sequences(cv2, np, torch, train_rows, report, args.width, args.height)
        print(
            f"device={device} train_rows={len(train_rows)} eval_rows={len(eval_rows)} train_sequences={len(sequences)}",
            flush=True,
        )
        history = train_adapter_distribution(torch, F, model, sequences, args, device)
    elif args.feature_loss:
        sequences = make_sequences(cv2, np, torch, train_rows, report, args.width, args.height)
        print(
            f"device={device} train_rows={len(train_rows)} eval_rows={len(eval_rows)} train_sequences={len(sequences)}",
            flush=True,
        )
        if not sequences:
            raise RuntimeError("no training sequences")
        history = train_adapter_feature(torch, F, model, sequences, args, device)
    else:
        items = make_items(cv2, np, torch, train_rows, report, args.width, args.height)
        print(f"device={device} train_rows={len(train_rows)} eval_rows={len(eval_rows)} train_items={len(items)}", flush=True)
        if not items:
            raise RuntimeError("no training items")
        history = train_adapter(torch, F, model, items, args, device)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out_dir / "adapter.pt")
    write_history(args.out_dir / "history.csv", history)
    infer_rows(cv2, np, torch, model, eval_rows, report, args.out_dir, device)


if __name__ == "__main__":
    main()
