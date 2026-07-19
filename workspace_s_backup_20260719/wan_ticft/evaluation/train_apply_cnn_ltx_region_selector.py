"""Train the final CNN selector and apply it to BDD test candidate frames."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb, write_rgb
from baseline.refiner_model import import_torch
from baseline.verify import verify_sample_output
from wan_ticft.evaluation.evaluate_cnn_ltx_region_selector import (
    build_model,
    make_input,
    predict_alpha,
    prepare_lowres,
    read_jsonl,
    train_fold,
)
from wan_ticft.evaluation.evaluate_video_reports import read_report


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--ltx_test_dir", type=Path, required=True)
    parser.add_argument("--region_test_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report_csv", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--estimate_width", type=int, default=640)
    parser.add_argument("--model_width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda_tv", type=float, default=0.01)
    parser.add_argument("--lambda_entropy", type=float, default=0.002)
    parser.add_argument("--alpha_strength", type=float, default=1.0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    torch, nn, F = import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reports = read_report(args.ltx_report)
    train_rows = [row for row in read_jsonl(args.manifest) if row["sample_id"] in reports]
    print(f"=== Prepare {len(train_rows)} selector training samples ===", flush=True)
    train = [
        sample
        for row in train_rows
        if (
            sample := prepare_lowres(
                cv2,
                np,
                row,
                reports[row["sample_id"]],
                args.estimate_width,
                args.width,
                args.height,
            )
        )
    ]
    model = train_fold(torch, nn, F, np, train, args, device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "width": args.width,
                "height": args.height,
                "model_width": args.model_width,
                "epochs": args.epochs,
            },
        },
        args.checkpoint,
    )
    del train

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    test_rows = [row for row in read_csv(args.test_csv) if row.get("view_type") == "bdd"]
    for position, row in enumerate(test_rows, start=1):
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        history_paths = numeric_pngs(Path(row["input_dir"]))
        region_paths = numeric_pngs(args.region_test_dir / output_name)
        ltx_paths = numeric_pngs(args.ltx_test_dir / output_name)
        if len(history_paths) < 2:
            raise RuntimeError(f"{output_name}: fewer than two history frames")
        if len(region_paths) != target_n or len(ltx_paths) != target_n:
            raise RuntimeError(
                f"{output_name}: candidate count region={len(region_paths)} ltx={len(ltx_paths)} expected={target_n}"
            )
        previous = read_rgb(cv2, history_paths[-2])
        last = read_rgb(cv2, history_paths[-1])
        if previous is None or last is None:
            raise RuntimeError(f"{output_name}: unreadable history")
        sample_out = args.out_dir / output_name
        sample_out.mkdir(parents=True, exist_ok=True)
        alpha_means = []
        alpha_hard = []
        for index, (region_path, ltx_path) in enumerate(zip(region_paths, ltx_paths)):
            region = read_rgb(cv2, region_path)
            ltx = read_rgb(cv2, ltx_path)
            if region is None or ltx is None:
                raise RuntimeError(f"{output_name}: unreadable candidates at {index}")
            alpha = predict_alpha(
                torch,
                np,
                cv2,
                model,
                device,
                previous,
                last,
                region,
                ltx,
                index,
                target_n,
                args.width,
                args.height,
            )
            if args.alpha_strength != 1.0:
                clipped = np.clip(alpha, 1e-4, 1.0 - 1e-4)
                logits = np.log(clipped / (1.0 - clipped)) * args.alpha_strength
                alpha = 1.0 / (1.0 + np.exp(-logits))
            output = np.clip(
                region.astype(np.float32) * (1.0 - alpha[:, :, None])
                + ltx.astype(np.float32) * alpha[:, :, None],
                0,
                255,
            ).astype(np.uint8)
            if not write_rgb(cv2, sample_out / f"{index}.png", output):
                raise RuntimeError(f"{output_name}: cannot write frame {index}")
            alpha_means.append(float(alpha.mean()))
            alpha_hard.append(float((alpha >= 0.5).mean()))

        verified = verify_sample_output(sample_out, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verification failed: {verified.notes}")
        report.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "target_N": target_n,
                "frames_found": verified.frames_found,
                "alpha_mean": f"{float(np.mean(alpha_means)):.6f}",
                "alpha_hard_fraction": f"{float(np.mean(alpha_hard)):.6f}",
                "status": "ok",
            }
        )
        print(
            f"[{position}/{len(test_rows)}] {output_name} | alpha={np.mean(alpha_means):.3f}",
            flush=True,
        )

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(report),
                "frames": sum(int(row["frames_found"]) for row in report),
                "alpha_mean": float(np.mean([float(row["alpha_mean"]) for row in report])),
                "alpha_strength": args.alpha_strength,
                "checkpoint": str(args.checkpoint),
                "status": "ok",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(report), "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
