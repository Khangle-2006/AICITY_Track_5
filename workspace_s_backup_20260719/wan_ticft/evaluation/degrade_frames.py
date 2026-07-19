"""Add real-camera characteristics (blur, sensor noise, JPEG, resolution softening) to
PNG-folder predictions to move synthetic-looking frames toward the real-image distribution.
Per-pixel/per-frame only -> geometry and temporal layout are preserved."""
from __future__ import annotations
import argparse, csv, shutil
from pathlib import Path
import cv2, numpy as np


def degrade(img, blur, noise, jpeg_q, downscale, rng):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    if downscale and downscale < 1.0:
        nw, nh = max(1, int(w * downscale)), max(1, int(h * downscale))
        out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    if blur and blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if noise and noise > 0:
        out = out + rng.normal(0.0, noise, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if jpeg_q and jpeg_q < 100:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_q)])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--blur", type=float, default=0.6)
    p.add_argument("--noise", type=float, default=4.0)
    p.add_argument("--jpeg_q", type=int, default=35)
    p.add_argument("--downscale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean", action="store_true")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    if a.clean and a.out_dir.exists():
        shutil.rmtree(a.out_dir)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    folders = [d for d in sorted(a.base_dir.iterdir()) if d.is_dir()]
    print(f"degrading {len(folders)} folders | blur={a.blur} noise={a.noise} jpeg_q={a.jpeg_q} downscale={a.downscale}")
    for i, d in enumerate(folders, 1):
        od = a.out_dir / d.name
        od.mkdir(parents=True, exist_ok=True)
        pngs = sorted(d.glob("*.png"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
        for png in pngs:
            img = cv2.imread(str(png))
            if img is None:
                raise RuntimeError(f"unreadable {png}")
            cv2.imwrite(str(od / png.name), degrade(img, a.blur, a.noise, a.jpeg_q, a.downscale, rng))
        print(f"[{i}/{len(folders)}] {d.name} | {len(pngs)} frames", flush=True)
    # copy report.csv if present (keeps build_hybrid_submission happy)
    rep = a.base_dir / "report.csv"
    if rep.exists():
        shutil.copy2(rep, a.out_dir / "report.csv")
    print(f"output={a.out_dir}")


if __name__ == "__main__":
    main()
