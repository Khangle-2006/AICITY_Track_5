"""Trustworthy FID: standard Inception FID (pytorch-fid) between a method's frames and REAL GT
frames (val). Optionally applies camera-degradation to candidate frames BEFORE FID, so we can
find the degradation that minimizes REAL-FID offline before submitting."""
from __future__ import annotations
import argparse, shutil
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


def gather_real(real_root: Path, work: Path, limit: int) -> set[str]:
    out = work / "real"; out.mkdir(parents=True, exist_ok=True)
    sids = []
    for d in sorted(real_root.iterdir()):
        if not d.is_dir() or "bdd" not in d.name:
            continue
        pngs = sorted((d / "target").glob("*.png")) if (d / "target").exists() else []
        if not pngs:
            continue
        sids.append(d.name)
        for p in pngs:
            shutil.copy2(p, out / f"{d.name}__{p.name}")
        if limit and len(sids) >= limit:
            break
    return set(sids)


def decode_candidate(name, vid_dir, work, sids, deg, rng):
    out = work / name; out.mkdir(parents=True, exist_ok=True)
    n = 0
    for vid in sorted(Path(vid_dir).glob("*")):
        if vid.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
            continue
        sid = vid.stem
        if sids and sid not in sids:
            continue
        cap = cv2.VideoCapture(str(vid)); idx = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if deg:
                fr = degrade(fr, *deg, rng)
            cv2.imwrite(str(out / f"{sid}__{idx:04d}.png"), fr); idx += 1; n += 1
        cap.release()
    return n


def fid(real_dir, cand_dir):
    from pytorch_fid.fid_score import calculate_fid_given_paths
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return calculate_fid_given_paths([str(real_dir), str(cand_dir)], 50, dev, 2048, 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real_root", type=Path, required=True)
    p.add_argument("--candidate", action="append", required=True, metavar="NAME=VIDEO_DIR")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--blur", type=float, default=0.0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--jpeg_q", type=int, default=100)
    p.add_argument("--downscale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    deg = None
    if a.blur or a.noise or a.jpeg_q < 100 or a.downscale < 1.0:
        deg = (a.blur, a.noise, a.jpeg_q, a.downscale)
    rng = np.random.default_rng(a.seed)
    if a.work.exists():
        shutil.rmtree(a.work)
    a.work.mkdir(parents=True, exist_ok=True)
    sids = gather_real(a.real_root, a.work, a.limit)
    tag = f"blur={a.blur} noise={a.noise} jpeg_q={a.jpeg_q} ds={a.downscale}" if deg else "RAW"
    print(f"real samples={len(sids)} | degrade: {tag}", flush=True)
    results = {}
    for spec in a.candidate:
        name, d = spec.split("=", 1)
        decode_candidate(name, d, a.work, sids, deg, rng)
        results[name] = fid(a.work / "real", a.work / name)
        print(f"  {name:18s} REAL-FID={results[name]:.3f}", flush=True)
    print("=== REAL-FID (lower=better) ===")
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {k:18s} {v:.3f}")


if __name__ == "__main__":
    main()
