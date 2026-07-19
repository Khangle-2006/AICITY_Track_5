from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix CNN-selector BDD test frames back toward LTX frames.")
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--cnn_dir", type=Path, required=True)
    parser.add_argument("--ltx_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ltx_mix", type=float, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.ltx_mix <= 1.0:
        raise ValueError("--ltx_mix must be in [0, 1]")

    cv2, np = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    rows = [row for row in read_csv(args.test_csv) if row.get("view_type") == "bdd"]
    for position, row in enumerate(rows, start=1):
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        cnn_paths = numeric_pngs(args.cnn_dir / output_name)
        ltx_paths = numeric_pngs(args.ltx_dir / output_name)
        if len(cnn_paths) != target_n or len(ltx_paths) != target_n:
            raise RuntimeError(
                f"{output_name}: candidate count cnn={len(cnn_paths)} ltx={len(ltx_paths)} expected={target_n}"
            )

        sample_out = args.out_dir / output_name
        sample_out.mkdir(parents=True, exist_ok=True)
        for index, (cnn_path, ltx_path) in enumerate(zip(cnn_paths, ltx_paths)):
            cnn = read_rgb(cv2, cnn_path)
            ltx = read_rgb(cv2, ltx_path)
            if cnn is None or ltx is None:
                raise RuntimeError(f"{output_name}: unreadable frame {index}")
            if cnn.shape[:2] != ltx.shape[:2]:
                ltx = cv2.resize(ltx, (cnn.shape[1], cnn.shape[0]), interpolation=cv2.INTER_CUBIC)
            mixed = np.clip(
                (1.0 - args.ltx_mix) * cnn.astype(np.float32) + args.ltx_mix * ltx.astype(np.float32),
                0,
                255,
            ).astype(np.uint8)
            if not write_rgb(cv2, sample_out / f"{index}.png", mixed):
                raise RuntimeError(f"{output_name}: cannot write frame {index}")

        verified = verify_sample_output(sample_out, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verification failed: {verified.notes}")
        report.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "target_N": target_n,
                "frames_found": verified.frames_found,
                "ltx_mix": f"{args.ltx_mix:.6f}",
                "status": "ok",
            }
        )
        print(f"[{position}/{len(rows)}] {output_name} | ok", flush=True)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    (args.out_dir / "summary.json").write_text(
        json.dumps({"rows": len(report), "ltx_mix": args.ltx_mix, "status": "ok"}, indent=2),
        encoding="utf-8",
    )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
