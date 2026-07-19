from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import read_report


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_video(cv2, path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames, fps


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def resize_like(cv2, image, reference):
    height, width = reference.shape[:2]
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def dis_flow(cv2, source_gray, target_gray):
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setUseSpatialPropagation(True)
    return estimator.calc(source_gray, target_gray, None)


def remap(cv2, np, image, flow):
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return cv2.remap(
        image,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )


def alignment_confidence(cv2, np, base, donor, flow_base_to_donor, flow_donor_to_base):
    reverse_at_base = remap(cv2, np, flow_donor_to_base, flow_base_to_donor)
    fb_error = np.linalg.norm(flow_base_to_donor + reverse_at_base, axis=2)
    fb_conf = np.exp(-fb_error / 2.5)

    aligned = remap(cv2, np, donor, flow_base_to_donor)
    base_low = cv2.GaussianBlur(base.astype(np.float32), (0, 0), 3.0)
    aligned_low = cv2.GaussianBlur(aligned.astype(np.float32), (0, 0), 3.0)
    photo_error = np.mean(np.abs(base_low - aligned_low), axis=2)
    photo_conf = np.exp(-photo_error / 22.0)

    confidence = np.clip(fb_conf * photo_conf, 0.0, 1.0)
    confidence = cv2.GaussianBlur(confidence.astype(np.float32), (0, 0), 1.2)
    return aligned, confidence


def transfer_luma_high_frequency(cv2, np, base, donor, alpha: float, sigma: float):
    base_gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    donor_gray = cv2.cvtColor(donor, cv2.COLOR_RGB2GRAY)
    flow_base_to_donor = dis_flow(cv2, base_gray, donor_gray)
    flow_donor_to_base = dis_flow(cv2, donor_gray, base_gray)
    aligned, confidence = alignment_confidence(
        cv2,
        np,
        base,
        donor,
        flow_base_to_donor,
        flow_donor_to_base,
    )

    base_ycc = cv2.cvtColor(base, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    donor_ycc = cv2.cvtColor(aligned, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    base_y = base_ycc[..., 0]
    donor_y = donor_ycc[..., 0]
    base_high = base_y - cv2.GaussianBlur(base_y, (0, 0), sigma)
    donor_high = donor_y - cv2.GaussianBlur(donor_y, (0, 0), sigma)

    # Replace only aligned luma detail. Low frequencies, color, and geometry remain from the base.
    detail_delta = np.clip(donor_high - base_high, -24.0, 24.0)
    base_ycc[..., 0] = np.clip(base_y + alpha * confidence * detail_delta, 0.0, 255.0)
    return cv2.cvtColor(base_ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB), confidence


def donor_at_time(donor_frames, donor_fps: float, base_index: int, base_fps: float):
    index = int(round(base_index * donor_fps / max(base_fps, 1e-6)))
    return donor_frames[min(max(index, 0), len(donor_frames) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Motion-locked high-frequency texture transfer smoke.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--skyreels_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--variants", default="", help="Optional comma-separated variant names.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"][: args.limit]
    base_report = read_report(args.base_report)
    sky_report = read_report(args.skyreels_report)
    rows = [row for row in rows if row["sample_id"] in base_report and row["sample_id"] in sky_report]
    if not rows:
        raise RuntimeError("no common BDD samples across manifest/base/SkyReels reports")

    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    selected_manifest = args.out_root / "selected_manifest.jsonl"
    with selected_manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    variants = {
        "lossless_control": ("control", 0.0, 0.0),
        "sky_yhf_a15": ("sky", 0.15, 2.0),
        "sky_yhf_a25": ("sky", 0.25, 2.0),
        "sky_yhf_a35": ("sky", 0.35, 2.0),
        "condition_yhf_a25": ("condition", 0.25, 2.0),
    }
    if args.variants:
        requested = [name.strip() for name in args.variants.split(",") if name.strip()]
        unknown = [name for name in requested if name not in variants]
        if unknown:
            raise ValueError(f"unknown variants: {', '.join(unknown)}")
        variants = {name: variants[name] for name in requested}
    reports = {label: [] for label in variants}

    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        base_frames, base_fps = read_video(cv2, base_report[sample_id])
        sky_frames, sky_fps = read_video(cv2, sky_report[sample_id])
        condition_paths = (row.get("frames") or {}).get("condition") or []
        condition = read_rgb(cv2, Path(condition_paths[-1])) if condition_paths else None
        if condition is None:
            raise RuntimeError(f"{sample_id}: cannot read last condition frame")

        confidence_rows = []
        for label, (donor_kind, alpha, sigma) in variants.items():
            output_frames = []
            mean_confidences = []
            for frame_index, base in enumerate(base_frames):
                if donor_kind == "control":
                    output_frames.append(base.copy())
                    mean_confidences.append(1.0)
                    continue
                if donor_kind == "sky":
                    donor = donor_at_time(sky_frames, sky_fps, frame_index, base_fps)
                else:
                    donor = condition
                donor = resize_like(cv2, donor, base)
                output, confidence = transfer_luma_high_frequency(cv2, np, base, donor, alpha, sigma)
                output_frames.append(output)
                mean_confidences.append(float(confidence.mean()))

            output_path = args.out_root / label / "videos" / f"{sample_id}.avi"
            write_video(cv2, output_frames, output_path, base_fps)
            mean_confidence = float(np.mean(mean_confidences))
            reports[label].append(
                {
                    "sample_id": sample_id,
                    "video": str(output_path),
                    "status": "ok",
                    "source_fps": f"{base_fps:.6f}",
                    "future_frames": len(output_frames),
                    "donor": donor_kind,
                    "alpha": f"{alpha:.3f}",
                    "sigma": f"{sigma:.3f}",
                    "mean_confidence": f"{mean_confidence:.6f}",
                }
            )
            confidence_rows.append((label, mean_confidence))
        print(f"[{position}/{len(rows)}] {sample_id} | {confidence_rows}", flush=True)

    for label, report_rows in reports.items():
        report_path = args.out_root / label / "report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
            writer.writeheader()
            writer.writerows(report_rows)
    print(f"output={args.out_root}", flush=True)


if __name__ == "__main__":
    main()
