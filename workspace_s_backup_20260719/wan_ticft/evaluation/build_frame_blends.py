from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb, write_rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--correction_dir", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--alphas", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    samples = sorted(
        path.name
        for path in args.correction_dir.iterdir()
        if path.is_dir() and (args.base_dir / path.name).is_dir()
    )
    summary = {"samples": samples, "alphas": args.alphas, "outputs": {}}

    for percent in args.alphas:
        alpha = percent / 100.0
        output = args.out_root / f"wan_router_{percent}"
        if args.clean and output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        written = 0
        for sample in samples:
            base_paths = numeric_pngs(args.base_dir / sample)
            correction_paths = numeric_pngs(args.correction_dir / sample)
            count = min(len(base_paths), len(correction_paths))
            sample_out = output / sample
            sample_out.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                base = read_rgb(cv2, base_paths[index])
                correction = read_rgb(cv2, correction_paths[index])
                if base is None or correction is None:
                    continue
                if correction.shape[:2] != base.shape[:2]:
                    correction = cv2.resize(
                        correction,
                        (base.shape[1], base.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                blended = np.clip(
                    (1.0 - alpha) * base.astype(np.float32)
                    + alpha * correction.astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)
                written += int(write_rgb(cv2, sample_out / f"{index}.png", blended))
        summary["outputs"][str(percent)] = {"path": str(output), "frames_written": written}
        print(f"alpha={percent}% samples={len(samples)} frames={written}", flush=True)

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

