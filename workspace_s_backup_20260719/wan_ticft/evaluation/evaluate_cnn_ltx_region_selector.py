"""Cross-validate a convolutional spatial selector between LTX and region predictions."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from baseline.refiner_model import import_torch
from wan_ticft.evaluation.evaluate_bdd_ego_motion import region_predictions
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_images(cv2, paths: list[str]):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def target_offsets(row: dict, count: int) -> list[int]:
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    target = indices.get("target") or []
    if not condition or len(target) != count:
        return list(range(1, count + 1))
    last = int(condition[-1])
    return [max(1, int(index) - last) for index in target]


def resize(cv2, image, width: int, height: int):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def resize_like(cv2, image, target):
    if image.shape[:2] == target.shape[:2]:
        return image
    return cv2.resize(image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)


def load_candidates(cv2, np, row, video, estimate_width):
    paths = row.get("frames") or {}
    conditions = load_images(cv2, paths.get("condition") or [])
    targets = load_images(cv2, paths.get("target") or [])
    if len(conditions) < 2 or not targets:
        return None
    indices = (row.get("frame_indices") or {}).get("condition") or []
    history_gap = max(1, int(indices[-1]) - int(indices[-2])) if len(indices) >= 2 else 1
    offsets = target_offsets(row, len(targets))
    region = region_predictions(cv2, np, conditions[-2], conditions[-1], offsets, history_gap, estimate_width)
    mapped = map_target_offsets(row, video_fps(cv2, video))
    frame_map = read_video_rgb_frames(cv2, video, mapped)
    ltx = [frame_map[index] for index in mapped if index in frame_map]
    if len(ltx) != len(targets):
        return None
    region = [resize_like(cv2, frame, target) for frame, target in zip(region, targets)]
    ltx = [resize_like(cv2, frame, target) for frame, target in zip(ltx, targets)]
    return conditions[-2], conditions[-1], region, ltx, targets


def make_input(np, previous, last, region, ltx, horizon: float):
    delta = np.abs(region.astype(np.float32) - ltx.astype(np.float32))
    horizon_map = np.full((*region.shape[:2], 1), horizon * 255.0, dtype=np.float32)
    return np.concatenate([previous, last, region, ltx, delta, horizon_map], axis=2).astype(np.float32) / 255.0


def prepare_lowres(cv2, np, row, video, estimate_width, width, height):
    loaded = load_candidates(cv2, np, row, video, estimate_width)
    if loaded is None:
        return None
    previous, last, region, ltx, targets = loaded
    previous = resize(cv2, previous, width, height)
    last = resize(cv2, last, width, height)
    count = len(targets)
    items = []
    for index, (region_frame, ltx_frame, target) in enumerate(zip(region, ltx, targets)):
        region_small = resize(cv2, region_frame, width, height)
        ltx_small = resize(cv2, ltx_frame, width, height)
        target_small = resize(cv2, target, width, height)
        items.append(
            (
                make_input(np, previous, last, region_small, ltx_small, index / max(1, count - 1)),
                region_small.astype(np.float32) / 255.0,
                ltx_small.astype(np.float32) / 255.0,
                target_small.astype(np.float32) / 255.0,
            )
        )
    return {"sample_id": row["sample_id"], "items": items}


def build_model(nn, channels: int, width: int):
    class Selector(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, width, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, 1, 3, padding=1),
            )

        def forward(self, inputs):
            return self.encoder(inputs).sigmoid()

    return Selector()


def tensorize(torch, np, item):
    x, region, ltx, target = item
    arrays = [torch.from_numpy(value.astype(np.float32)).permute(2, 0, 1) for value in (x, region, ltx, target)]
    return arrays


def train_fold(torch, nn, F, np, samples, args, device):
    model = build_model(nn, 16, args.model_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    items = [item for sample in samples for item in sample["items"]]
    rng = random.Random(17)
    for epoch in range(args.epochs):
        rng.shuffle(items)
        model.train()
        total = 0.0
        count = 0
        for start in range(0, len(items), args.batch_size):
            batch = [tensorize(torch, np, item) for item in items[start : start + args.batch_size]]
            x = torch.stack([item[0] for item in batch]).to(device)
            region = torch.stack([item[1] for item in batch]).to(device)
            ltx = torch.stack([item[2] for item in batch]).to(device)
            target = torch.stack([item[3] for item in batch]).to(device)
            alpha = model(x)
            pred = region * (1.0 - alpha) + ltx * alpha
            reconstruction = F.l1_loss(pred, target)
            tv = (alpha[:, :, :, 1:] - alpha[:, :, :, :-1]).abs().mean()
            tv += (alpha[:, :, 1:, :] - alpha[:, :, :-1, :]).abs().mean()
            entropy = (alpha * (1.0 - alpha)).mean()
            loss = reconstruction + args.lambda_tv * tv + args.lambda_entropy * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
        print(f"epoch {epoch + 1}/{args.epochs} | loss={total / max(count, 1):.6f}", flush=True)
    return model


def predict_alpha(torch, np, cv2, model, device, previous, last, region, ltx, index, count, width, height):
    previous = resize(cv2, previous, width, height)
    last = resize(cv2, last, width, height)
    region_small = resize(cv2, region, width, height)
    ltx_small = resize(cv2, ltx, width, height)
    inputs = make_input(np, previous, last, region_small, ltx_small, index / max(1, count - 1))
    x = torch.from_numpy(inputs).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        alpha = model(x)[0, 0].detach().cpu().numpy()
    return cv2.resize(alpha, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_CUBIC).clip(0.0, 1.0)


def score_frames(cv2, np, frames, targets):
    values = [(psnr(cv2, np, p, t), ssim_gray(cv2, np, p, t), mean_absdiff(np, p, t)) for p, t in zip(frames, targets)]
    return tuple(float(np.mean([value[i] for value in values])) for i in range(3))


def sharpen_alpha(np, cv2, alpha, strength: float, *, hard: bool = False):
    if hard:
        binary = (alpha >= 0.5).astype(np.float32)
        return cv2.GaussianBlur(binary, (0, 0), sigmaX=0.8, sigmaY=0.8).clip(0.0, 1.0)
    clipped = np.clip(alpha, 1e-4, 1.0 - 1e-4)
    logits = np.log(clipped / (1.0 - clipped)) * strength
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def blend_candidates(np, region, ltx, alpha):
    return np.clip(
        region.astype(np.float32) * (1.0 - alpha[:, :, None])
        + ltx.astype(np.float32) * alpha[:, :, None],
        0,
        255,
    ).astype(np.uint8)


def evaluate_sample(torch, np, cv2, model, device, row, video, args):
    previous, last, region, ltx, targets = load_candidates(cv2, np, row, video, args.estimate_width)
    cnn, sharp2, sharp4, hard_feather, oracle = [], [], [], [], []
    for index, (r, l, target) in enumerate(zip(region, ltx, targets)):
        alpha = predict_alpha(torch, np, cv2, model, device, previous, last, r, l, index, len(targets), args.width, args.height)
        cnn.append(blend_candidates(np, r, l, alpha))
        sharp2.append(blend_candidates(np, r, l, sharpen_alpha(np, cv2, alpha, 2.0)))
        sharp4.append(blend_candidates(np, r, l, sharpen_alpha(np, cv2, alpha, 4.0)))
        hard_feather.append(blend_candidates(np, r, l, sharpen_alpha(np, cv2, alpha, 1.0, hard=True)))
        r_error = np.mean(np.abs(r.astype(np.float32) - target.astype(np.float32)), axis=2)
        l_error = np.mean(np.abs(l.astype(np.float32) - target.astype(np.float32)), axis=2)
        oracle.append(np.where((l_error < r_error)[:, :, None], l, r))
    return {
        "region": region,
        "ltx16": ltx,
        "cnn_selector": cnn,
        "cnn_sharp2": sharp2,
        "cnn_sharp4": sharp4,
        "cnn_hard_feather": hard_feather,
        "pixel_oracle": oracle,
    }, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--estimate_width", type=int, default=640)
    parser.add_argument("--model_width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda_tv", type=float, default=0.01)
    parser.add_argument("--lambda_entropy", type=float, default=0.002)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    torch, nn, F = import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports = read_report(args.ltx_report)
    rows = [row for row in read_jsonl(args.manifest) if row["sample_id"] in reports]
    folds = [(rows[::2], rows[1::2]), (rows[1::2], rows[::2])]
    results = []
    for fold, (train_rows, test_rows) in enumerate(folds, start=1):
        print(f"=== fold {fold}: prepare {len(train_rows)} train samples ===", flush=True)
        train = [
            sample
            for row in train_rows
            if (sample := prepare_lowres(cv2, np, row, reports[row["sample_id"]], args.estimate_width, args.width, args.height))
        ]
        model = train_fold(torch, nn, F, np, train, args, device)
        del train
        for position, row in enumerate(test_rows, start=1):
            candidates, targets = evaluate_sample(torch, np, cv2, model, device, row, reports[row["sample_id"]], args)
            for candidate, frames in candidates.items():
                p, s, d = score_frames(cv2, np, frames, targets)
                results.append({"fold": fold, "sample_id": row["sample_id"], "candidate": candidate, "psnr": f"{p:.6f}", "ssim": f"{s:.6f}", "absdiff": f"{d:.6f}"})
            print(f"fold {fold} [{position}/{len(test_rows)}] {row['sample_id']}", flush=True)
        del model
        torch.cuda.empty_cache()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"{'Candidate':<20} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 60)
    for candidate in ("region", "ltx16", "cnn_selector", "cnn_sharp2", "cnn_sharp4", "cnn_hard_feather", "pixel_oracle"):
        items = [row for row in results if row["candidate"] == candidate]
        print(f"{candidate:<20} {len(items):>5} {np.mean([float(x['psnr']) for x in items]):>10.4f} {np.mean([float(x['ssim']) for x in items]):>10.4f} {np.mean([float(x['absdiff']) for x in items]):>10.4f}")
    print(f"output {args.out_csv}")


if __name__ == "__main__":
    main()
