from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output
from wan_ticft.evaluation.train_temporal_latent_refiner import (
    build_model,
    image_tensor,
    import_torch,
    resize,
    tensor_to_rgb,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_state_dict(raw: dict) -> dict:
    state = raw.get("model", raw)
    if any(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def make_input_sequence(torch, np, cv2, base_frames, condition_frames, width: int, height: int):
    last = resize(cv2, condition_frames[-1], width, height)
    prev = resize(cv2, condition_frames[-2], width, height) if len(condition_frames) >= 2 else last
    history_delta = (last.astype(np.float32) - prev.astype(np.float32)) / 255.0
    xs = []
    base_low = []
    count = len(base_frames)
    prev_base = None
    for index, base in enumerate(base_frames):
        base_s = resize(cv2, base, width, height)
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
        base_low.append(image_tensor(torch, np, base_s))
        prev_base = base_s
    return torch.stack(xs), torch.stack(base_low)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply trained temporal latent refiner to BDD test frame folders.")
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    torch, nn, F = import_torch()
    cv2, np = import_cv2_np()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    width = int(ckpt_args.get("width", 320))
    height = int(ckpt_args.get("height", 180))
    model_width = int(ckpt_args.get("model_width", 64))
    residual_scale = float(ckpt_args.get("residual_scale", 0.06))

    model = build_model(nn, in_channels=13, width=model_width, residual_scale=residual_scale).to(device)
    model.load_state_dict(load_state_dict(checkpoint), strict=True)
    model.eval()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for row in read_csv(args.test_csv) if row.get("view_type") == "bdd"]
    if args.limit:
        rows = rows[: args.limit]
    report = []

    with torch.no_grad():
        for position, row in enumerate(rows, start=1):
            output_name = row["source_folder"]
            sample_id = row["sample_id"]
            target_n = int(row["target_N"])
            base_paths = numeric_pngs(args.base_dir / output_name)
            condition_paths = numeric_pngs(Path(row["input_dir"]))
            if len(base_paths) != target_n:
                raise RuntimeError(f"{output_name}: base frames={len(base_paths)} expected={target_n}")
            if len(condition_paths) < 2:
                raise RuntimeError(f"{output_name}: need at least 2 condition frames")

            base_frames = [read_rgb(cv2, path) for path in base_paths]
            condition_frames = [read_rgb(cv2, path) for path in condition_paths[-2:]]
            if any(frame is None for frame in base_frames) or any(frame is None for frame in condition_frames):
                raise RuntimeError(f"{output_name}: unreadable frame")

            x, base_low = make_input_sequence(torch, np, cv2, base_frames, condition_frames, width, height)
            pred_low = model(x.unsqueeze(0).to(device))[0]

            sample_out = args.out_dir / output_name
            if sample_out.exists():
                shutil.rmtree(sample_out)
            sample_out.mkdir(parents=True)
            for index, base in enumerate(base_frames):
                residual = (pred_low[index] - base_low[index].to(device)).unsqueeze(0)
                residual_up = F.interpolate(residual, size=base.shape[:2], mode="bilinear", align_corners=False)[0]
                base_tensor = image_tensor(torch, np, base).to(device)
                pred = (base_tensor + residual_up).clamp(0, 1)
                if not write_rgb(cv2, sample_out / f"{index}.png", tensor_to_rgb(np, pred)):
                    raise RuntimeError(f"{output_name}: cannot write frame {index}")

            verified = verify_sample_output(sample_out, target_n)
            if not verified.ok:
                raise RuntimeError(f"{output_name}: verification failed: {verified.notes}")
            report.append(
                {
                    "output_name": output_name,
                    "sample_id": sample_id,
                    "target_N": target_n,
                    "frames_found": verified.frames_found,
                    "checkpoint": str(args.checkpoint),
                    "status": "ok",
                }
            )
            print(f"[{position}/{len(rows)}] {output_name} | ok", flush=True)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
