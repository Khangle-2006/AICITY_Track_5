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
        description="Composite train/val retrieval background with test object-layer foreground."
    )
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--matches_csv", required=True)
    parser.add_argument("--object_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--route_csv", required=True)
    parser.add_argument("--sheets_dir", default="")
    parser.add_argument("--mode", choices=sorted(MODES), default="strict")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--delta_threshold", type=float, default=16.0)
    parser.add_argument("--foreign_threshold", type=float, default=24.0)
    parser.add_argument("--mask_blur", type=float, default=4.0)
    parser.add_argument("--dilate", type=int, default=11)
    return parser


def read_index(index_dir: Path) -> list[dict[str, str]]:
    with (index_dir / "test_index.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_best_matches(path: Path) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            best.setdefault(row["query_sample_id"], row)
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


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return image


def resize_like(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def read_reference_frame(
    cap: cv2.VideoCapture,
    frame_index: int,
    orientation: str,
    width: int,
    height: int,
) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
    ok, frame = cap.read()
    if not ok:
        return None
    frame = ORIENTATIONS[orientation](frame)
    return resize_like(frame, width, height)


def motion_mask(
    current: np.ndarray,
    previous: np.ndarray,
    threshold: float,
    dilate: int,
    blur: float,
) -> np.ndarray:
    diff = np.mean(cv2.absdiff(current, previous), axis=2)
    mask = (diff > threshold).astype(np.uint8) * 255
    if dilate > 1:
        kernel = np.ones((dilate, dilate), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
    alpha = mask.astype(np.float32) / 255.0
    if blur > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=blur)
    return np.clip(alpha[..., None], 0.0, 1.0)


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


def composite_retrieval_background(
    row: dict[str, str],
    match: dict[str, str],
    object_sample_dir: Path,
    out_sample_dir: Path,
    args: argparse.Namespace,
) -> tuple[int, str]:
    target_n = int(row["target_N"])
    width = int(row["frame_width"])
    height = int(row["frame_height"])
    input_paths = numeric_pngs(Path(row["input_dir"]))
    if not input_paths:
        return 0, "missing_input_history"
    query_last = resize_like(read_image(input_paths[-1]), width, height)

    cap = cv2.VideoCapture(match["reference_video"])
    if not cap.isOpened():
        return 0, "cannot_open_reference"

    end_frame = int(match["reference_end_frame"])
    stride = int(match["source_stride"])
    orientation = match["orientation"]
    prev_retrieval = read_reference_frame(cap, end_frame, orientation, width, height)
    if prev_retrieval is None:
        cap.release()
        return 0, "cannot_read_ref_last"

    if out_sample_dir.exists():
        shutil.rmtree(out_sample_dir)
    out_sample_dir.mkdir(parents=True, exist_ok=True)

    prev_object = query_last
    frames_written = 0
    last_retrieval = prev_retrieval
    for output_index in range(target_n):
        retrieval = read_reference_frame(
            cap,
            end_frame + (output_index + 1) * stride,
            orientation,
            width,
            height,
        )
        if retrieval is None:
            retrieval = last_retrieval.copy()
        object_frame = resize_like(read_image(object_sample_dir / f"{output_index}.png"), width, height)

        retrieval_dynamic = motion_mask(
            retrieval,
            last_retrieval,
            args.delta_threshold,
            args.dilate,
            args.mask_blur,
        )
        retrieval_foreign = motion_mask(
            retrieval,
            query_last,
            args.foreign_threshold,
            args.dilate,
            args.mask_blur,
        )
        object_dynamic = motion_mask(
            object_frame,
            prev_object,
            args.delta_threshold,
            args.dilate,
            args.mask_blur,
        )
        mask = np.maximum(np.maximum(retrieval_dynamic, retrieval_foreign), object_dynamic)
        composite = retrieval.astype(np.float32) * (1.0 - mask) + object_frame.astype(np.float32) * mask
        cv2.imwrite(str(out_sample_dir / f"{output_index}.png"), np.clip(composite, 0, 255).astype(np.uint8))

        prev_object = object_frame
        last_retrieval = retrieval
        frames_written += 1
    cap.release()
    return frames_written, "ok"


def make_sheet(row: dict[str, str], object_dir: Path, out_dir: Path, sheet_path: Path) -> None:
    input_paths = numeric_pngs(Path(row["input_dir"]))
    if not input_paths:
        return
    target_n = int(row["target_N"])
    picks = [0, max(0, target_n // 2), max(0, target_n - 1)]
    tiles: list[np.ndarray] = []

    def add(label: str, image: np.ndarray | None) -> None:
        if image is None:
            return
        tile = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
        tiles.append(tile)

    add("query_last", read_image(input_paths[-1]))
    for pick in picks:
        add(f"comp_{pick}", cv2.imread(str(out_dir / f"{pick}.png")))
    for pick in picks:
        add(f"object_{pick}", cv2.imread(str(object_dir / f"{pick}.png")))
    if not tiles:
        return
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[index : index + 4]) for index in range(0, len(tiles), 4)]
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sheet_path), np.vstack(rows))


def main() -> None:
    args = build_parser().parse_args()
    index_dir = Path(args.index_dir)
    matches = read_best_matches(Path(args.matches_csv))
    object_dir = Path(args.object_dir)
    out_dir = Path(args.out_dir)
    route_csv = Path(args.route_csv)
    sheets_dir = Path(args.sheets_dir) if args.sheets_dir else None

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    route_csv.parent.mkdir(parents=True, exist_ok=True)

    route_rows: list[dict[str, object]] = []
    rows = [row for row in read_index(index_dir) if row["view_type"].startswith("wts_")]
    for row in rows:
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        object_sample_dir = object_dir / output_name
        destination = out_dir / output_name
        match = matches.get(row["sample_id"])
        use_retrieval, reason = should_use_retrieval(row, match, args.mode)
        if use_retrieval and match is not None:
            frames, status = composite_retrieval_background(row, match, object_sample_dir, destination, args)
            if frames != target_n or status != "ok":
                frames = copy_fallback(object_sample_dir, destination)
                source_choice = "fallback_after_composite_fail"
                reason = status
            else:
                source_choice = "retrieval_background_object_layer"
                if sheets_dir is not None:
                    make_sheet(row, object_sample_dir, destination, sheets_dir / f"{output_name}.jpg")
        else:
            frames = copy_fallback(object_sample_dir, destination)
            source_choice = "fallback_object_layer"

        verified = verify_sample_output(destination, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verify failed: {verified.notes}")

        route_rows.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "view_type": row["view_type"],
                "target_N": target_n,
                "source_choice": source_choice,
                "reason": reason,
                "frames_found": verified.frames_found,
                "reference_split": (match or {}).get("reference_split", ""),
                "reference_video": (match or {}).get("reference_video", ""),
                "reference_end_frame": (match or {}).get("reference_end_frame", ""),
                "source_stride": (match or {}).get("source_stride", ""),
                "sequence_mad": (match or {}).get("sequence_mad", ""),
                "orb_inlier_ratio": (match or {}).get("orb_inlier_ratio", ""),
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

    print(f"mode={args.mode}")
    print(f"out_dir={out_dir}")
    print(f"route_csv={route_csv}")
    print(f"rows={len(route_rows)}")
    print(f"counts={dict(Counter(row['source_choice'] for row in route_rows))}")
    if sheets_dir is not None:
        print(f"sheets={sheets_dir}")


if __name__ == "__main__":
    main()
