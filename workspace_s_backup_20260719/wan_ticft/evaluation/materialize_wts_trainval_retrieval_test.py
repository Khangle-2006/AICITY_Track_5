from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from baseline.frame_io import numeric_pngs
from baseline.verify import verify_sample_output


ORIENTATIONS = {
    "none": lambda image: image,
    "cw": lambda image: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
    "ccw": lambda image: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    "180": lambda image: cv2.rotate(image, cv2.ROTATE_180),
}

MODES = {
    "ultra": {"max_sequence_mad": 6.0, "min_orb_inlier_ratio": 0.97},
    "strict": {"max_sequence_mad": 12.0, "min_orb_inlier_ratio": 0.90},
    "loose": {"max_sequence_mad": 21.0, "min_orb_inlier_ratio": 0.86},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use train/val nearest-neighbor continuations for high-confidence WTS overhead test samples."
    )
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--matches_csv", required=True)
    parser.add_argument("--fallback_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--route_csv", required=True)
    parser.add_argument("--sheets_dir", default="")
    parser.add_argument("--mode", choices=sorted(MODES), default="strict")
    parser.add_argument("--clean", action="store_true")
    return parser


def read_index(index_dir: Path) -> list[dict[str, str]]:
    with (index_dir / "test_index.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_best_matches(path: Path) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = row["query_sample_id"]
            if sample_id not in best:
                best[sample_id] = row
    return best


def copy_fallback(source_dir: Path, destination: Path) -> int:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for png in numeric_pngs(source_dir):
        shutil.copy2(png, destination / f"{int(png.stem)}.png")
        count += 1
    return count


def extract_retrieval(
    match: dict[str, str],
    out_dir: Path,
    target_n: int,
    width: int,
    height: int,
) -> tuple[int, str]:
    video_path = Path(match["reference_video"])
    end_frame = int(match["reference_end_frame"])
    stride = int(match["source_stride"])
    orientation = match["orientation"]
    transform = ORIENTATIONS[orientation]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, f"cannot_open_reference:{video_path}"

    frames_written = 0
    last_frame: np.ndarray | None = None
    for output_index in range(target_n):
        source_index = end_frame + (output_index + 1) * stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = cap.read()
        if not ok:
            if last_frame is None:
                cap.release()
                return frames_written, f"cannot_read_reference_frame:{source_index}"
            frame = last_frame.copy()
        else:
            frame = transform(frame)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            last_frame = frame.copy()
        cv2.imwrite(str(out_dir / f"{output_index}.png"), frame)
        frames_written += 1
    cap.release()
    return frames_written, "ok"


def make_sheet(
    row: dict[str, str],
    match: dict[str, str],
    output_dir: Path,
    fallback_dir: Path,
    sheet_path: Path,
) -> None:
    input_paths = numeric_pngs(Path(row["input_dir"]))
    if not input_paths:
        return
    query_last = cv2.imread(str(input_paths[-1]))
    ref_last = None
    cap = cv2.VideoCapture(match["reference_video"])
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(match["reference_end_frame"]))
        ok, ref_last = cap.read()
        if ok:
            ref_last = ORIENTATIONS[match["orientation"]](ref_last)
    cap.release()

    def read_output_frame(directory: Path, index: int) -> np.ndarray | None:
        return cv2.imread(str(directory / f"{index}.png"))

    target_n = int(row["target_N"])
    picks = [0, max(0, target_n // 2), max(0, target_n - 1)]
    tiles: list[np.ndarray] = []
    labels = []
    images = [query_last, ref_last]
    labels.extend(["query_last", "ref_last"])
    for pick in picks:
        images.append(read_output_frame(output_dir, pick))
        labels.append(f"retr_{pick}")
    for pick in picks:
        images.append(read_output_frame(fallback_dir, pick))
        labels.append(f"base_{pick}")

    for label, image in zip(labels, images):
        if image is None:
            continue
        image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
        tiles.append(image)
    if not tiles:
        return
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[index : index + 4]) for index in range(0, len(tiles), 4)]
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sheet_path), np.vstack(rows))


def should_use_retrieval(row: dict[str, str], match: dict[str, str] | None, mode: str) -> tuple[bool, str]:
    if row["view_type"] != "wts_overhead_or_fixed_camera":
        return False, "not_overhead"
    if not match:
        return False, "no_match"
    if match["reference_split"] not in {"train", "val"}:
        return False, "reference_not_trainval"
    cfg = MODES[mode]
    sequence_mad = float(match["sequence_mad"])
    orb = float(match["orb_inlier_ratio"])
    if sequence_mad > cfg["max_sequence_mad"]:
        return False, f"sequence_mad>{cfg['max_sequence_mad']:g}"
    if orb < cfg["min_orb_inlier_ratio"]:
        return False, f"orb<{cfg['min_orb_inlier_ratio']:g}"
    return True, "pass"


def main() -> None:
    args = build_parser().parse_args()

    index_dir = Path(args.index_dir)
    matches_csv = Path(args.matches_csv)
    fallback_dir = Path(args.fallback_dir)
    out_dir = Path(args.out_dir)
    route_csv = Path(args.route_csv)
    sheets_dir = Path(args.sheets_dir) if args.sheets_dir else None

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    route_csv.parent.mkdir(parents=True, exist_ok=True)

    matches = read_best_matches(matches_csv)
    rows = [row for row in read_index(index_dir) if row["view_type"].startswith("wts_")]
    route_rows: list[dict[str, object]] = []

    for row in rows:
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        fallback_sample = fallback_dir / output_name
        destination = out_dir / output_name
        use_retrieval, reason = should_use_retrieval(row, matches.get(row["sample_id"]), args.mode)

        if use_retrieval:
            match = matches[row["sample_id"]]
            frames, status = extract_retrieval(
                match,
                destination,
                target_n,
                int(row["frame_width"]),
                int(row["frame_height"]),
            )
            if frames != target_n or status != "ok":
                copy_fallback(fallback_sample, destination)
                source_choice = "fallback_after_extract_fail"
                reason = status
            else:
                source_choice = "trainval_retrieval"
                if sheets_dir is not None:
                    make_sheet(row, match, destination, fallback_sample, sheets_dir / f"{output_name}.jpg")
        else:
            frames = copy_fallback(fallback_sample, destination)
            source_choice = "fallback_object_layer"

        verified = verify_sample_output(destination, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verify failed after route: {verified.notes}")

        match = matches.get(row["sample_id"], {})
        route_rows.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "view_type": row["view_type"],
                "target_N": target_n,
                "source_choice": source_choice,
                "reason": reason,
                "frames_found": verified.frames_found,
                "reference_split": match.get("reference_split", ""),
                "reference_video": match.get("reference_video", ""),
                "reference_end_frame": match.get("reference_end_frame", ""),
                "source_stride": match.get("source_stride", ""),
                "sequence_mad": match.get("sequence_mad", ""),
                "orb_inlier_ratio": match.get("orb_inlier_ratio", ""),
                "status": "ok",
            }
        )

    fields = [
        "output_name",
        "sample_id",
        "view_type",
        "target_N",
        "source_choice",
        "reason",
        "frames_found",
        "reference_split",
        "reference_video",
        "reference_end_frame",
        "source_stride",
        "sequence_mad",
        "orb_inlier_ratio",
        "status",
    ]
    with route_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for route_row in route_rows:
            writer.writerow({field: route_row.get(field, "") for field in fields})

    counts = Counter(row["source_choice"] for row in route_rows)
    print(f"mode={args.mode}")
    print(f"out_dir={out_dir}")
    print(f"route_csv={route_csv}")
    print(f"rows={len(route_rows)}")
    print(f"counts={dict(counts)}")
    if sheets_dir is not None:
        print(f"sheets={sheets_dir}")


if __name__ == "__main__":
    main()
