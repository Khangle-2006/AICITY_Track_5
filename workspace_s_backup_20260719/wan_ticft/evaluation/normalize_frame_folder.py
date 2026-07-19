from __future__ import annotations

import argparse
import collections
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize all PNGs in a folder to one image size.")
    parser.add_argument("--in_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, _np = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    before: collections.Counter[tuple[int, int] | None] = collections.Counter()
    written = 0
    for path in sorted(args.in_dir.glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        before[None if image is None else image.shape[:2]] += 1
        if image is None:
            continue
        if image.shape[:2] != (args.height, args.width):
            image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_CUBIC)
        if cv2.imwrite(str(args.out_dir / path.name), image):
            written += 1
    print(f"before={dict(before)}")
    print(f"written={written} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
