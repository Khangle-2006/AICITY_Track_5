from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np
from wan_ticft.evaluation.evaluate_cnn_ltx_region_selector import (
    blend_candidates,
    evaluate_sample,
    prepare_lowres,
    read_jsonl,
    sharpen_alpha,
    train_fold,
)
from wan_ticft.evaluation.evaluate_video_reports import read_report


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def video_fps(cv2, path: Path, default: float = 30.0) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return default
    fps = float(cap.get(cv2.CAP_PROP_FPS) or default)
    cap.release()
    return fps


def parse_variants(values: list[str]) -> list[tuple[str, str, float]]:
    variants = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError(f"variant must be name:kind:value, got {value!r}")
        name, kind, raw = parts
        if kind not in {"alpha_strength", "ltx_mix"}:
            raise ValueError(f"unknown variant kind {kind!r}")
        variants.append((name, kind, float(raw)))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize CNN selector variants for BDD validation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--estimate_width", type=int, default=640)
    parser.add_argument("--model_width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda_tv", type=float, default=0.01)
    parser.add_argument("--lambda_entropy", type=float, default=0.002)
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
    )
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    from baseline.refiner_model import import_torch

    torch, nn, F = import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports = read_report(args.ltx_report)
    rows = [row for row in read_jsonl(args.manifest) if row["sample_id"] in reports]
    default_variants = [
        "cnn:ltx_mix:0.0",
        "ltxmix25:ltx_mix:0.25",
        "ltxmix50:ltx_mix:0.50",
        "alpha_s05:alpha_strength:0.50",
        "alpha_s025:alpha_strength:0.25",
    ]
    variants = parse_variants(args.variant or default_variants)

    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    report_rows = {name: [] for name, _, _ in variants}
    folds = [(rows[::2], rows[1::2]), (rows[1::2], rows[::2])]
    for fold, (train_rows, test_rows) in enumerate(folds, start=1):
        print(f"=== fold {fold}: train={len(train_rows)} test={len(test_rows)} ===", flush=True)
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
        del train

        for position, row in enumerate(test_rows, start=1):
            source_video = reports[row["sample_id"]]
            fps = video_fps(cv2, source_video)
            candidates, _ = evaluate_sample(torch, np, cv2, model, device, row, source_video, args)
            region = candidates["region"]
            ltx = candidates["ltx16"]
            cnn = candidates["cnn_selector"]

            # Reuse the already materialized alpha variants from evaluate_sample when possible;
            # otherwise apply a conservative global mix back toward LTX.
            frames_by_variant = {}
            for name, kind, value in variants:
                if kind == "ltx_mix":
                    frames_by_variant[name] = [
                        np.clip((1.0 - value) * c.astype(np.float32) + value * l.astype(np.float32), 0, 255).astype(
                            np.uint8
                        )
                        for c, l in zip(cnn, ltx)
                    ]
                else:
                    # Approximate alpha softening by blending the CNN output toward the midpoint
                    # between CNN and LTX. Lower alpha strength means more LTX-like distribution.
                    mix = max(0.0, min(1.0, 1.0 - value))
                    frames_by_variant[name] = [
                        np.clip((1.0 - mix) * c.astype(np.float32) + mix * l.astype(np.float32), 0, 255).astype(
                            np.uint8
                        )
                        for c, l in zip(cnn, ltx)
                    ]

            for name, frames in frames_by_variant.items():
                path = args.out_root / name / "videos" / f"{row['sample_id']}.mp4"
                # Video-report evaluation samples future offsets 1..N. Store a
                # duplicate index-0 frame so offsets 1..N map to the N
                # predicted future frames instead of missing the last one.
                video_frames = [frames[0], *frames]
                write_video(cv2, video_frames, path, fps)
                report_rows[name].append(
                    {
                        "sample_id": row["sample_id"],
                        "video": str(path),
                        "status": "ok",
                        "source_fps": f"{fps:.6f}",
                        "future_frames": len(video_frames),
                    }
                )
            print(f"fold {fold} [{position}/{len(test_rows)}] {row['sample_id']} | ok", flush=True)
        del model
        torch.cuda.empty_cache()

    for name, rows_out in report_rows.items():
        path = args.out_root / name / "report.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows_out[0].keys())
            writer.writeheader()
            writer.writerows(rows_out)

    summary = {"rows": len(rows), "variants": [name for name, _, _ in variants], "status": "ok"}
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
