"""TEST multi-frame depth fusion + optional chunk residual-fill. Warp multiple observed frames
into each target (anchor on top, older history fills disocclusions with real pixels). Residual
holes (no history coverage) optionally filled by chunk (coherent structure) instead of base.
Writes PNG-per-output_name for build_hybrid_submission."""
from __future__ import annotations
import os as _os  # _CPU_CAP: avoid hogging all server cores
_T=_os.environ.get("CPU_THREADS","4")
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v,_T)
import argparse, csv, shutil
from pathlib import Path
import cv2, numpy as np, torch
from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb, write_rgb
import cv2 as _cv2; _cv2.setNumThreads(int(_T))
from baseline.verify import verify_sample_output
from wan_ticft.evaluation.materialize_depth_layered_renderer import (
    PerceptionModels, camera_matrix, compose, estimate_pose, render_confidence, smooth_pose_sequence, zbuffer_splat,
)
from wan_ticft.evaluation.materialize_depth_layered_multiframe import compose_pose, IDENTITY


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--base_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model_cache", type=Path, required=True)
    p.add_argument("--chunk_dir", type=Path, default=None, help="optional: fill residual holes with these frames")
    p.add_argument("--hole_cov", type=float, default=0.5, help="coverage below this = residual hole")
    p.add_argument("--view_type", default="bdd")
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--max_history", type=int, default=8)
    p.add_argument("--base_mode", choices=["video","repeat_anchor"], default="video")
    p.add_argument("--history_stride", type=int, default=2)
    p.add_argument("--focal_scale", type=float, default=0.9)
    p.add_argument("--point_stride", type=int, default=2)
    p.add_argument("--splat_radius", type=int, default=1)
    p.add_argument("--min_matches", type=int, default=20)
    p.add_argument("--min_inliers", type=int, default=12)
    p.add_argument("--photo_sigma", type=float, default=32.0)
    p.add_argument("--segment_batch", type=int, default=4)
    p.add_argument("--pose_smoothing_window", type=int, default=7)
    p.add_argument("--filled_pose_scale", type=float, default=0.35)
    p.add_argument("--confidence_ema", type=float, default=0.65)
    p.add_argument("--clean", action="store_true")
    a = p.parse_args()

    cv2m, _ = import_cv2_np()
    if a.clean and a.out_dir.exists():
        shutil.rmtree(a.out_dir)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    models = PerceptionModels(torch.device("cuda"), a.model_cache, depth_backend="dpt")
    rows = [r for r in read_csv(a.test_csv) if r.get("view_type") == a.view_type]
    report = []
    for pos, row in enumerate(rows, 1):
        out_name = row["source_folder"]; target_n = int(row["target_N"])
        cond_paths = numeric_pngs(Path(row["input_dir"]))
        bases = [read_rgb(cv2m, q) for q in numeric_pngs(a.base_dir / out_name)]
        bases = [b for b in bases if b is not None]
        conditions = [read_rgb(cv2m, q) for q in cond_paths]
        conditions = [c for c in conditions if c is not None]
        if not conditions or len(bases) != target_n:
            raise RuntimeError(f"{out_name}: conditions={len(conditions)} bases={len(bases)} target={target_n}")
        h, w = bases[0].shape[:2]
        bases = [cv2m.resize(b, (w, h), interpolation=cv2m.INTER_AREA) for b in bases]
        chunks = None
        if a.chunk_dir is not None:
            cf = [read_rgb(cv2m, q) for q in numeric_pngs(a.chunk_dir / out_name)]
            cf = [c for c in cf if c is not None]
            if len(cf) == target_n:
                chunks = [cv2m.resize(c, (w, h), interpolation=cv2m.INTER_AREA) for c in cf]
        sel = conditions[-(a.max_history * a.history_stride):][:: a.history_stride]
        if sel[-1] is not conditions[-1]:
            sel.append(conditions[-1])
        hist = [cv2m.resize(c, (w, h), interpolation=cv2m.INTER_AREA) for c in sel]
        anchor = hist[-1]
        fillbase = anchor if a.base_mode == "repeat_anchor" else None
        intrinsic = camera_matrix(w, h, a.focal_scale)
        depths = [models.depth(hf) for hf in hist]
        hactors = models.actor_masks(hist, a.segment_batch)
        statics = [(ha < 0.1).astype(np.float32) for ha in hactors]
        to_anchor = []
        for i, hf in enumerate(hist):
            if i == len(hist) - 1:
                to_anchor.append(IDENTITY)
            else:
                ph, _ = estimate_pose(hf, anchor, depths[i], hactors[i], intrinsic, a.min_matches, a.min_inliers)
                to_anchor.append(ph)
        bmasks = models.actor_masks(bases, a.segment_batch)
        raw, quals = [], []
        for b in bases:
            pa, _ = estimate_pose(anchor, b, depths[-1], hactors[-1], intrinsic, a.min_matches, a.min_inliers)
            raw.append(pa); quals.append(0.0 if pa is None else 1.0)
        poses, pquals, _ = smooth_pose_sequence(raw, quals, a.pose_smoothing_window, a.filled_pose_scale)

        sample_out = a.out_dir / out_name
        sample_out.mkdir(parents=True, exist_ok=True)
        prev_conf = None
        for t, base in enumerate(bases):
            pose_at = poses[t]; q = pquals[t]
            if pose_at is None or q <= 0:
                out = fillbase if fillbase is not None else base; prev_conf = None
            else:
                fused = (fillbase if fillbase is not None else base).copy(); cov_acc = np.zeros((h, w), np.float32)
                for i, hf in enumerate(hist):
                    pi = to_anchor[i]
                    if pi is None:
                        continue
                    pose_it = compose_pose(pi, pose_at)
                    rendered_i, cov_i = zbuffer_splat(hf, depths[i], statics[i], pose_it, intrinsic, a.point_stride, a.splat_radius)
                    m = cov_i > 0.5
                    fused[m] = rendered_i[m]
                    cov_acc = np.maximum(cov_acc, cov_i)
                base_eff = fillbase if fillbase is not None else base
                if chunks is not None:
                    hole = (cov_acc < a.hole_cov) & (bmasks[t] < 0.1)
                    if hole.any():
                        base_eff = base.copy(); base_eff[hole] = chunks[t][hole]
                conf = render_confidence(base_eff, fused, cov_acc, bmasks[t], q, a.photo_sigma, prev_conf, a.confidence_ema)
                out = compose(base_eff, fused, conf, a.alpha); prev_conf = conf
            if not write_rgb(cv2m, sample_out / f"{t}.png", out):
                raise RuntimeError(f"cannot write {sample_out}/{t}.png")
        v = verify_sample_output(sample_out, target_n)
        if not v.ok:
            raise RuntimeError(f"{out_name}: verify failed {v.notes}")
        report.append({"output_name": out_name, "sample_id": row["sample_id"], "view_type": row["view_type"],
                       "target_N": target_n, "frames_found": v.frames_found, "status": "ok"})
        ch = "chunk" if chunks is not None else "nochunk"
        print(f"[{pos}/{len(rows)}] {out_name} | ok (hist={len(hist)}, {ch})", flush=True)
    with (a.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as hd:
        wr = csv.DictWriter(hd, fieldnames=list(report[0])); wr.writeheader(); wr.writerows(report)
    print(f"output={a.out_dir}")


if __name__ == "__main__":
    main()
