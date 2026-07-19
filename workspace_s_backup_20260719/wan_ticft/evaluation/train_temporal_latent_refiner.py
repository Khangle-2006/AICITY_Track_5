from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import map_target_offsets, read_report, read_video_rgb_frames, video_fps
from wan_ticft.evaluation.train_metric_residual_adapter import ssim_loss, write_jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def resize(cv2, img, width: int, height: int):
    if img.shape[1] == width and img.shape[0] == height:
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def image_tensor(torch, np, img):
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def tensor_to_rgb(np, tensor):
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def load_targets(cv2, row: dict[str, Any]):
    return [img for path in (row.get("frames") or {}).get("target", []) if (img := read_rgb(cv2, Path(path))) is not None]


def load_condition(cv2, row: dict[str, Any]):
    paths = (row.get("frames") or {}).get("condition", [])
    imgs = [img for path in paths if (img := read_rgb(cv2, Path(path))) is not None]
    return imgs


def load_base_frames(cv2, row: dict[str, Any], video: Path):
    offsets = map_target_offsets(row, video_fps(cv2, video))
    frame_map = read_video_rgb_frames(cv2, video, offsets)
    return [frame_map[index] for index in offsets if index in frame_map]


def make_sequence(torch, np, cv2, row: dict[str, Any], base_video: Path, width: int, height: int):
    bases = load_base_frames(cv2, row, base_video)
    targets = load_targets(cv2, row)
    conditions = load_condition(cv2, row)
    if len(bases) != len(targets) or not bases or not conditions:
        return None
    last = resize(cv2, conditions[-1], width, height)
    prev = resize(cv2, conditions[-2], width, height) if len(conditions) >= 2 else last
    history_delta = (last.astype(np.float32) - prev.astype(np.float32)) / 255.0
    xs, ys, base_low = [], [], []
    count = len(bases)
    prev_base = None
    for index, (base, target) in enumerate(zip(bases, targets)):
        base_s = resize(cv2, base, width, height)
        target_s = resize(cv2, target, width, height)
        if prev_base is None:
            base_delta = history_delta
        else:
            base_delta = (base_s.astype(np.float32) - prev_base.astype(np.float32)) / 255.0
        t = 0.0 if count <= 1 else index / float(count - 1)
        time = np.full((height, width, 1), t, dtype=np.float32)
        x = np.concatenate(
            [
                base_s.astype(np.float32) / 255.0,
                last.astype(np.float32) / 255.0,
                base_delta.astype(np.float32),
                history_delta.astype(np.float32),
                time,
            ],
            axis=2,
        )
        xs.append(torch.from_numpy(x).permute(2, 0, 1).float())
        ys.append(image_tensor(torch, np, target_s))
        base_low.append(image_tensor(torch, np, base_s))
        prev_base = base_s
    return {"x": torch.stack(xs), "y": torch.stack(ys), "base": torch.stack(base_low), "row": row, "base_video": base_video}


def build_model(nn, in_channels: int, width: int, residual_scale: float):
    class ResidualBlock3D(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
                nn.SiLU(inplace=True),
                nn.Conv3d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(x + self.net(x))

    class TemporalLatentRefiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.residual_scale = residual_scale
            self.stem = nn.Sequential(nn.Conv3d(in_channels, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(inplace=True))
            self.body = nn.Sequential(*[ResidualBlock3D(width) for _ in range(5)])
            self.head = nn.Sequential(nn.Conv3d(width, width, 3, padding=1), nn.SiLU(inplace=True), nn.Conv3d(width, 3, 3, padding=1), nn.Tanh())

        def forward(self, x):
            # x: B,T,C,H,W
            x = x.permute(0, 2, 1, 3, 4)
            base = x[:, :3]
            residual = self.head(self.body(self.stem(x))) * self.residual_scale
            return (base + residual).clamp(0, 1).permute(0, 2, 1, 3, 4)

    return TemporalLatentRefiner()


def temporal_loss(pred, target):
    return ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs().mean()


def residual_temporal_loss(pred, base):
    residual = pred - base
    return (residual[:, 1:] - residual[:, :-1]).abs().mean()


def train(torch, F, model, sequences, args, device):
    image_model = None
    video_model = None
    image_mean = image_std = video_mean = video_std = None
    if args.lambda_image_feature > 0:
        from torchvision.models import Inception_V3_Weights, inception_v3

        image_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        image_model.fc = torch.nn.Identity()
        image_model.eval().to(device)
        for p in image_model.parameters():
            p.requires_grad_(False)
        if args.data_parallel and torch.cuda.device_count() > 1:
            image_model = torch.nn.DataParallel(image_model)
        image_mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
        image_std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    if args.lambda_video_feature > 0:
        from torchvision.models.video import R3D_18_Weights, r3d_18

        video_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        video_model.fc = torch.nn.Identity()
        video_model.eval().to(device)
        for p in video_model.parameters():
            p.requires_grad_(False)
        if args.data_parallel and torch.cuda.device_count() > 1:
            video_model = torch.nn.DataParallel(video_model)
        video_mean = torch.tensor((0.43216, 0.394666, 0.37645), device=device)[None, :, None, None, None]
        video_std = torch.tensor((0.22803, 0.22145, 0.216989), device=device)[None, :, None, None, None]

    def image_features(x):
        chunks = []
        chunk_size = args.feature_chunk_size or x.shape[0]
        for part in x.split(chunk_size, dim=0):
            part = F.interpolate(part, size=(299, 299), mode="bilinear", align_corners=False)
            chunks.append(image_model((part - image_mean) / image_std))
        return torch.cat(chunks, dim=0)

    def video_features(x):
        # x: B,T,C,H,W -> B,C,T,H,W
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w)
        flat = F.interpolate(flat, size=(112, 112), mode="bilinear", align_corners=False)
        video = flat.reshape(b, t, c, 112, 112).permute(0, 2, 1, 3, 4)
        return video_model((video - video_mean) / video_std)

    rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(sequences)
        model.train()
        totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "temp": 0.0, "res": 0.0, "res_temp": 0.0, "img": 0.0, "vid": 0.0}
        count = 0
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start : start + args.batch_size]
            x = torch.stack([item["x"] for item in batch]).to(device)
            y = torch.stack([item["y"] for item in batch]).to(device)
            base = torch.stack([item["base"] for item in batch]).to(device)
            pred = model(x)
            b, t, c, h, w = pred.shape
            pred_flat = pred.reshape(b * t, c, h, w)
            y_flat = y.reshape(b * t, c, h, w)
            l1 = (pred_flat - y_flat).abs().mean()
            sl = ssim_loss(torch, F, pred_flat, y_flat)
            tl = temporal_loss(pred, y)
            rl = (pred - base).abs().mean()
            rtl = residual_temporal_loss(pred, base)
            img = pred_flat.new_tensor(0.0)
            vid = pred_flat.new_tensor(0.0)
            if args.lambda_image_feature > 0:
                with torch.no_grad():
                    y_img = image_features(y_flat)
                img = (image_features(pred_flat) - y_img).abs().mean()
            if args.lambda_video_feature > 0:
                with torch.no_grad():
                    y_vid = video_features(y)
                vid = (video_features(pred) - y_vid).abs().mean()
            loss = (
                args.lambda_l1 * l1
                + args.lambda_ssim * sl
                + args.lambda_temporal * tl
                + args.lambda_residual * rl
                + args.lambda_residual_temporal * rtl
                + args.lambda_image_feature * img
                + args.lambda_video_feature * vid
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bs = len(batch)
            totals["loss"] += float(loss.detach().cpu()) * bs
            totals["l1"] += float(l1.detach().cpu()) * bs
            totals["ssim"] += float(sl.detach().cpu()) * bs
            totals["temp"] += float(tl.detach().cpu()) * bs
            totals["res"] += float(rl.detach().cpu()) * bs
            totals["res_temp"] += float(rtl.detach().cpu()) * bs
            totals["img"] += float(img.detach().cpu()) * bs
            totals["vid"] += float(vid.detach().cpu()) * bs
            count += bs
        row = {"epoch": epoch, **{key: value / max(1, count) for key, value in totals.items()}}
        history.append(row)
        print(
            f"epoch={epoch} loss={row['loss']:.6f} l1={row['l1']:.6f} ssim={row['ssim']:.6f} "
            f"temp={row['temp']:.6f} res={row['res']:.6f} res_temp={row['res_temp']:.6f} "
            f"img={row['img']:.6f} vid={row['vid']:.6f}",
            flush=True,
        )
    return history


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "loss", "l1", "ssim", "temp", "res", "res_temp", "img", "vid"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_video(cv2, frames, path: Path, fps: float, lossless: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    codec = "FFV1" if lossless else "mp4v"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    # Generated-video reports reserve index 0, while target offsets start at 1.
    # Duplicate the first prediction so frames 1..N map to predictions 0..N-1.
    for frame in [frames[0], *frames]:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def infer(torch, F, np, cv2, model, eval_sequences, args, out_dir: Path, device):
    rows = []
    model.eval()
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for pos, item in enumerate(eval_sequences, start=1):
            row = item["row"]
            sample_id = row["sample_id"]
            base_video = item["base_video"]
            x = item["x"].unsqueeze(0).to(device)
            pred_low = model(x)[0]
            base_frames = load_base_frames(cv2, row, base_video)
            frames = []
            for index, base in enumerate(base_frames):
                base_low = item["base"][index].to(device)
                residual = (pred_low[index] - base_low).unsqueeze(0)
                residual_up = F.interpolate(residual, size=base.shape[:2], mode="bilinear", align_corners=False)[0]
                base_tensor = image_tensor(torch, np, base).to(device)
                pred = (base_tensor + residual_up).clamp(0, 1)
                frames.append(tensor_to_rgb(np, pred))
            suffix = ".avi" if args.lossless_output else ".mp4"
            path = video_dir / f"{sample_id}{suffix}"
            write_video(cv2, frames, path, video_fps(cv2, base_video) or 30.0, args.lossless_output)
            rows.append({"sample_id": sample_id, "video": str(path), "status": "ok", "future_frames": len(frames)})
            print(f"infer [{pos}/{len(eval_sequences)}] {sample_id} | ok", flush=True)
    with (out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "future_frames"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sequence-level low-res temporal latent refiner over base report.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--train_manifest", type=Path, default=None)
    parser.add_argument("--train_base_report", type=Path, default=None)
    parser.add_argument("--eval_manifest", type=Path, default=None)
    parser.add_argument("--eval_base_report", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--eval_limit", type=int, default=0)
    parser.add_argument("--train_rows", type=int, default=16)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--model_width", type=int, default=64)
    parser.add_argument("--residual_scale", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_l1", type=float, default=0.45)
    parser.add_argument("--lambda_ssim", type=float, default=0.20)
    parser.add_argument("--lambda_temporal", type=float, default=0.20)
    parser.add_argument("--lambda_residual", type=float, default=0.12)
    parser.add_argument("--lambda_residual_temporal", type=float, default=0.25)
    parser.add_argument("--lambda_image_feature", type=float, default=0.0)
    parser.add_argument("--lambda_video_feature", type=float, default=0.0)
    parser.add_argument("--feature_chunk_size", type=int, default=8)
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--infer_all", action="store_true")
    parser.add_argument("--lossless_output", action="store_true")
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    torch, nn, F = import_torch()
    cv2, np = import_cv2_np()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args_json = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.out_dir / "args.json").write_text(json.dumps(args_json, indent=2), encoding="utf-8")

    def build_sequences(manifest: Path, report_path: Path, limit: int) -> list[dict[str, Any]]:
        report = read_report(report_path)
        rows = [row for row in read_jsonl(manifest) if row.get("view_type") == "bdd"]
        if limit:
            rows = rows[:limit]
        out = []
        for row in rows:
            video = report.get(row["sample_id"])
            if video is None:
                continue
            seq = make_sequence(torch, np, cv2, row, video, args.width, args.height)
            if seq is not None:
                out.append(seq)
        return out

    if args.train_manifest or args.eval_manifest:
        train_manifest = args.train_manifest or args.manifest
        train_report = args.train_base_report or args.base_report
        eval_manifest = args.eval_manifest or args.manifest
        eval_report = args.eval_base_report or args.base_report
        train_limit = args.train_limit or args.limit
        eval_limit = args.eval_limit or args.limit
        train_sequences_all = build_sequences(train_manifest, train_report, train_limit)
        train_sequences = train_sequences_all[: args.train_rows] if args.train_rows else train_sequences_all
        eval_sequences = build_sequences(eval_manifest, eval_report, eval_limit)
    else:
        sequences = build_sequences(args.manifest, args.base_report, args.limit)
        train_sequences = sequences[: args.train_rows]
        eval_sequences = sequences if args.infer_all else sequences[args.train_rows :]
    if not train_sequences or not eval_sequences:
        raise RuntimeError("need non-empty train/eval sequences")
    write_jsonl(args.out_dir / "train_manifest.jsonl", [item["row"] for item in train_sequences])
    write_jsonl(args.out_dir / "eval_manifest.jsonl", [item["row"] for item in eval_sequences])
    print(
        f"device={device} train_sequences={len(train_sequences)} eval_sequences={len(eval_sequences)} "
        f"size={args.width}x{args.height}",
        flush=True,
    )
    model = build_model(nn, in_channels=13, width=args.model_width, residual_scale=args.residual_scale).to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = torch.nn.DataParallel(model)
    history = train(torch, F, model, train_sequences, args, device)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out_dir / "temporal_latent_refiner.pt")
    write_history(args.out_dir / "history.csv", history)
    infer(torch, F, np, cv2, model, eval_sequences, args, args.out_dir, device)


if __name__ == "__main__":
    main()
