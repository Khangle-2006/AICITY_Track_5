from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .console import kv, section
from .frame_io import import_cv2_np
from .metrics import mean_absdiff, psnr, ssim_gray
from .refiner_dataset import RefinerDataset, filter_rows, load_split_rows, sample_rows
from .refiner_features import build_feature_config, refiner_in_channels
from .refiner_model import TinyResidualRefiner, import_torch, refiner_loss


def collate_skip_errors(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    from torch.utils.data._utils.collate import default_collate

    return default_collate(valid)


def make_loader(torch, dataset, batch_size: int, shuffle: bool, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_skip_errors,
    )


def batch_to_device(batch, device):
    return batch["x"].to(device), batch["base"].to(device), batch["target"].to(device)


def train_one_epoch(torch, model, loader, optimizer, device, args) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "l1": 0.0, "grad": 0.0, "residual": 0.0}
    n = 0
    for step, batch in enumerate(loader):
        if batch is None:
            continue
        x, base, target = batch_to_device(batch, device)
        pred = model(x, base)
        loss, parts = refiner_loss(torch, pred, target, base, args.lambda_grad, args.lambda_residual)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        totals["loss"] += float(loss.detach().cpu()) * bs
        for k, v in parts.items():
            totals[k] += v * bs
        n += bs
        if args.max_steps_per_epoch and step + 1 >= args.max_steps_per_epoch:
            break
    return {k: v / max(n, 1) for k, v in totals.items()}


def tensor_to_rgb(np, tensor):
    arr = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def evaluate(torch, np, cv2, model, loader, device, args) -> dict[str, float]:
    model.eval()
    totals = {
        "model_psnr": 0.0,
        "model_ssim": 0.0,
        "model_absdiff": 0.0,
        "base_psnr": 0.0,
        "base_ssim": 0.0,
        "base_absdiff": 0.0,
    }
    n = 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if batch is None:
                continue
            x, base, target = batch_to_device(batch, device)
            pred = model(x, base)
            for i in range(x.shape[0]):
                pred_img = tensor_to_rgb(np, pred[i])
                base_img = tensor_to_rgb(np, base[i])
                target_img = tensor_to_rgb(np, target[i])
                totals["model_psnr"] += psnr(cv2, np, pred_img, target_img)
                totals["model_ssim"] += ssim_gray(cv2, np, pred_img, target_img)
                totals["model_absdiff"] += mean_absdiff(np, pred_img, target_img)
                totals["base_psnr"] += psnr(cv2, np, base_img, target_img)
                totals["base_ssim"] += ssim_gray(cv2, np, base_img, target_img)
                totals["base_absdiff"] += mean_absdiff(np, base_img, target_img)
                n += 1
            if args.max_eval_batches and step + 1 >= args.max_eval_batches:
                break
    out = {k: v / max(n, 1) for k, v in totals.items()}
    out["n"] = float(n)
    out["d_psnr"] = out["model_psnr"] - out["base_psnr"]
    out["d_ssim"] = out["model_ssim"] - out["base_ssim"]
    out["d_absdiff"] = out["model_absdiff"] - out["base_absdiff"]
    return out


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "train_l1",
        "train_grad",
        "train_residual",
        "model_psnr",
        "base_psnr",
        "d_psnr",
        "model_ssim",
        "base_ssim",
        "d_ssim",
        "model_absdiff",
        "base_absdiff",
        "d_absdiff",
        "n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--out_dir", default="/workspace/experiments/refiner_baseline")
    parser.add_argument("--dataset_part", default="")
    parser.add_argument("--view_type", default="")
    parser.add_argument("--phase", default="")
    parser.add_argument("--max_train_rows", type=int, default=1000)
    parser.add_argument("--max_val_rows", type=int, default=250)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--target_gap_frames", type=int, default=4)
    parser.add_argument("--eval_horizon", type=int, default=4)
    parser.add_argument("--random_horizon", action="store_true")
    parser.add_argument("--horizon_min", type=int, default=1)
    parser.add_argument("--horizon_max", type=int, default=16)
    parser.add_argument("--time_normalized_features", action="store_true")
    parser.add_argument("--include_context_features", action="store_true")
    parser.add_argument("--max_context_horizon", type=int, default=120)
    parser.add_argument("--max_context_history_gap", type=int, default=30)
    parser.add_argument("--train_width", type=int, default=320)
    parser.add_argument("--train_height", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model_width", type=int, default=48)
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--lambda_grad", type=float, default=0.10)
    parser.add_argument("--lambda_residual", type=float, default=0.02)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch, _, _ = import_torch()
    cv2, np = import_cv2_np()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_dir = Path(args.index_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_cfg = build_feature_config(args)

    train_rows = filter_rows(load_split_rows(index_dir, "train"), args.dataset_part, args.view_type, args.phase)
    val_rows = filter_rows(load_split_rows(index_dir, "val"), args.dataset_part, args.view_type, args.phase)
    train_rows = sample_rows(train_rows, args.max_train_rows, args.seed)
    val_rows = sample_rows(val_rows, args.max_val_rows, args.seed + 1)

    train_ds = RefinerDataset(cv2, np, torch, train_rows, args, feature_cfg, train=True)
    val_ds = RefinerDataset(cv2, np, torch, val_rows, args, feature_cfg, train=False)
    train_loader = make_loader(torch, train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(torch, val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TinyResidualRefiner.build(
        in_channels=refiner_in_channels(feature_cfg),
        width=args.model_width,
        residual_scale=args.residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    section("TRAINABLE RESIDUAL FLOW REFINER")
    kv("index_dir", index_dir)
    kv("out_dir", out_dir)
    kv("device", device)
    kv("train rows", len(train_rows))
    kv("val rows", len(val_rows))
    kv("size", f"{args.train_width}x{args.train_height}")
    kv("epochs", args.epochs)
    kv("batch_size", args.batch_size)
    kv("target_gap_frames", args.target_gap_frames)
    kv("eval_horizon", args.eval_horizon)
    kv("random_horizon", args.random_horizon)
    if args.random_horizon:
        kv("horizon range", f"{args.horizon_min}..{args.horizon_max}")

    args_path = out_dir / "args.json"
    args_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    history = []
    best_score = None
    best_path = out_dir / "refiner_best.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(torch, model, train_loader, optimizer, device, args)
        val_metrics = evaluate(torch, np, cv2, model, val_loader, device, args)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_l1": train_metrics["l1"],
            "train_grad": train_metrics["grad"],
            "train_residual": train_metrics["residual"],
            **val_metrics,
        }
        history.append(row)
        write_history(out_dir / "history.csv", history)

        section(f"EPOCH {epoch}")
        kv("train_loss", f"{row['train_loss']:.6f}")
        kv("base_psnr", f"{row['base_psnr']:.4f}")
        kv("model_psnr", f"{row['model_psnr']:.4f}")
        kv("d_psnr", f"{row['d_psnr']:+.4f}")
        kv("base_ssim", f"{row['base_ssim']:.4f}")
        kv("model_ssim", f"{row['model_ssim']:.4f}")
        kv("d_ssim", f"{row['d_ssim']:+.4f}")
        kv("base_absdiff", f"{row['base_absdiff']:.4f}")
        kv("model_absdiff", f"{row['model_absdiff']:.4f}")
        kv("d_absdiff", f"{row['d_absdiff']:+.4f}")

        score = row["model_psnr"] + 10.0 * row["model_ssim"] - 0.10 * row["model_absdiff"]
        if best_score is None or score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "score": score}, best_path)

    torch.save({"model": model.state_dict(), "args": vars(args), "epoch": args.epochs}, out_dir / "refiner_last.pt")
    section("OUTPUT")
    kv("history", out_dir / "history.csv")
    kv("best checkpoint", best_path)
    kv("last checkpoint", out_dir / "refiner_last.pt")


if __name__ == "__main__":
    main()
