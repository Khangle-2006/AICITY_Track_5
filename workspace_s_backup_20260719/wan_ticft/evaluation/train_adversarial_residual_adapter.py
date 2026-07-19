from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np
from wan_ticft.evaluation.evaluate_video_reports import read_report
from wan_ticft.evaluation.train_metric_residual_adapter import (
    build_model,
    infer_rows,
    make_sequences,
    ssim_loss,
    write_history,
    write_jsonl,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def build_image_discriminator(nn):
    return nn.Sequential(
        nn.Conv2d(3, 32, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(32, 64, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 1, 3, padding=1),
    )


def build_video_discriminator(nn):
    return nn.Sequential(
        nn.Conv3d(3, 32, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv3d(32, 64, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv3d(64, 128, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv3d(128, 1, 3, padding=1),
    )


def sample_clip(torch, rng, sequences, clip_frames: int, sequence_batch: int, device):
    batch = []
    for _ in range(sequence_batch):
        seq = rng.choice(sequences)
        start = rng.randint(0, len(seq) - clip_frames)
        batch.append(seq[start : start + clip_frames])
    x = torch.stack([torch.stack([item["x"] for item in clip]) for clip in batch]).to(device)
    y = torch.stack([torch.stack([item["y"] for item in clip]) for clip in batch]).to(device)
    b = torch.stack([torch.stack([item["base"] for item in clip]) for clip in batch]).to(device)
    return x, y, b


def gan_d_loss(F, real_logits, fake_logits):
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def gan_g_loss(F, fake_logits):
    return F.softplus(-fake_logits).mean()


def train_gan(torch, nn, F, generator, sequences, args, device):
    valid = [seq for seq in sequences if len(seq) >= args.clip_frames]
    if not valid:
        raise RuntimeError("no valid train sequences")

    image_d = build_image_discriminator(nn).to(device)
    video_d = build_video_discriminator(nn).to(device)
    g_opt = torch.optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    d_opt = torch.optim.AdamW(
        list(image_d.parameters()) + list(video_d.parameters()),
        lr=args.discriminator_lr,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed)
    history = []

    for epoch in range(1, args.epochs + 1):
        generator.train()
        image_d.train()
        video_d.train()
        totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "temp": 0.0, "res": 0.0, "img_gan": 0.0, "vid_gan": 0.0, "dist": 0.0}
        steps = args.max_steps or max(1, len(valid) * 4)
        for _ in range(steps):
            x, y, base = sample_clip(torch, rng, valid, args.clip_frames, args.sequence_batch, device)
            batch, frames, channels, height, width = x.shape

            # Discriminator update.
            with torch.no_grad():
                fake = generator(x.reshape(batch * frames, channels, height, width)).reshape(batch, frames, 3, height, width)
            real_frames = y.reshape(batch * frames, 3, height, width)
            fake_frames = fake.reshape(batch * frames, 3, height, width)
            real_video = y.permute(0, 2, 1, 3, 4)
            fake_video = fake.permute(0, 2, 1, 3, 4)
            d_loss = gan_d_loss(F, image_d(real_frames), image_d(fake_frames.detach()))
            d_loss = d_loss + gan_d_loss(F, video_d(real_video), video_d(fake_video.detach()))
            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()

            # Generator update.
            pred = generator(x.reshape(batch * frames, channels, height, width)).reshape(batch, frames, 3, height, width)
            pred_frames = pred.reshape(batch * frames, 3, height, width)
            target_frames = y.reshape(batch * frames, 3, height, width)
            base_frames = base.reshape(batch * frames, 3, height, width)
            pred_video = pred.permute(0, 2, 1, 3, 4)
            l1 = (pred_frames - target_frames).abs().mean()
            sl = ssim_loss(torch, F, pred_frames, target_frames)
            temp = ((pred[:, 1:] - pred[:, :-1]) - (y[:, 1:] - y[:, :-1])).abs().mean()
            res = (pred_frames - base_frames).abs().mean()
            img_gan = gan_g_loss(F, image_d(pred_frames))
            vid_gan = gan_g_loss(F, video_d(pred_video))
            loss = (
                args.lambda_l1 * l1
                + args.lambda_ssim * sl
                + args.lambda_temporal * temp
                + args.lambda_residual * res
                + args.lambda_image_gan * img_gan
                + args.lambda_video_gan * vid_gan
            )
            g_opt.zero_grad(set_to_none=True)
            loss.backward()
            g_opt.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["l1"] += float(l1.detach().cpu())
            totals["ssim"] += float(sl.detach().cpu())
            totals["temp"] += float(temp.detach().cpu())
            totals["res"] += float(res.detach().cpu())
            totals["img_gan"] += float(img_gan.detach().cpu())
            totals["vid_gan"] += float(vid_gan.detach().cpu())
            totals["dist"] += float(d_loss.detach().cpu())

        row = {"epoch": epoch, **{key: value / steps for key, value in totals.items()}}
        history.append(row)
        print(
            f"epoch={epoch} loss={row['loss']:.6f} l1={row['l1']:.6f} ssim={row['ssim']:.6f} "
            f"temp={row['temp']:.6f} res={row['res']:.6f} "
            f"img_gan={row['img_gan']:.6f} vid_gan={row['vid_gan']:.6f} d={row['dist']:.6f}",
            flush=True,
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train adversarial residual adapter for FID/FVD-oriented refinement.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--train_rows", type=int, default=16)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--model_width", type=int, default=48)
    parser.add_argument("--residual_scale", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--clip_frames", type=int, default=8)
    parser.add_argument("--sequence_batch", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--discriminator_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_l1", type=float, default=0.20)
    parser.add_argument("--lambda_ssim", type=float, default=0.08)
    parser.add_argument("--lambda_temporal", type=float, default=0.08)
    parser.add_argument("--lambda_residual", type=float, default=0.18)
    parser.add_argument("--lambda_image_gan", type=float, default=0.04)
    parser.add_argument("--lambda_video_gan", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=41)
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
        raise RuntimeError("need non-empty train/eval split")

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train_manifest.jsonl", train_rows)
    write_jsonl(args.out_dir / "eval_manifest.jsonl", eval_rows)
    args_json = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.out_dir / "args.json").write_text(json.dumps(args_json, indent=2), encoding="utf-8")

    report = read_report(args.base_report)
    sequences = make_sequences(cv2, np, torch, train_rows, report, args.width, args.height)
    print(
        f"device={device} train_rows={len(train_rows)} eval_rows={len(eval_rows)} train_sequences={len(sequences)}",
        flush=True,
    )
    generator = build_model(nn, 7, args.model_width, args.residual_scale).to(device)
    history = train_gan(torch, nn, F, generator, sequences, args, device)
    torch.save({"model": generator.state_dict(), "args": vars(args)}, args.out_dir / "adapter.pt")
    write_history(args.out_dir / "history.csv", history)
    infer_rows(cv2, np, torch, generator, eval_rows, report, args.out_dir, device)


if __name__ == "__main__":
    main()
