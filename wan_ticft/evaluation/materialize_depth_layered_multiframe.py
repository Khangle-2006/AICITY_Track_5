"""Multi-frame depth-layered renderer: warp MULTIPLE history frames into each base/target view
and z-composite (anchor on top, older history fills disocclusions with REAL pixels). Fixes the
single-anchor 'drag/stretch' at disocclusions. Inference-only; never uses targets."""
from __future__ import annotations
import os as _os  # _CPU_CAP: avoid hogging all server cores
_T=_os.environ.get("CPU_THREADS","4")
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v,_T)
import argparse, csv, shutil
from pathlib import Path
import cv2, numpy as np, torch

from baseline.frame_io import import_cv2_np, read_rgb
import cv2 as _cv2; _cv2.setNumThreads(int(_T))
from wan_ticft.evaluation.evaluate_video_reports import map_target_offsets, read_report, read_video_rgb_frames, video_fps
from wan_ticft.evaluation.materialize_depth_layered_renderer import (
    PerceptionModels, camera_matrix, compose, estimate_pose, render_confidence,
    smooth_pose_sequence, zbuffer_splat, load_images, load_base_frames, load_full_video_frames,
    write_video, read_jsonl,
)


def compose_pose(p1, p2):
    """p1: h_i->anchor, p2: anchor->target  =>  h_i->target."""
    r1, t1 = p1; r2, t2 = p2
    R1 = cv2.Rodrigues(np.asarray(r1, dtype=np.float64))[0]
    R2 = cv2.Rodrigues(np.asarray(r2, dtype=np.float64))[0]
    R = R2 @ R1
    t = (R2 @ np.asarray(t1, dtype=np.float64).reshape(3, 1) + np.asarray(t2, dtype=np.float64).reshape(3, 1)).reshape(3)
    return (cv2.Rodrigues(R)[0], t)


IDENTITY = (np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--base_report", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--model_cache", type=Path, required=True)
    p.add_argument("--view_type", default="bdd")
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--max_history", type=int, default=4, help="history frames fused (incl anchor)")
    p.add_argument("--base_mode", choices=["video","repeat_anchor"], default="video", help="repeat_anchor = use real anchor frozen as base (no ghost, no synthetic)")
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
    p.add_argument("--preserve_full_video", action="store_true")
    p.add_argument("--clean", action="store_true")
    a = p.parse_args()

    cv2m, _ = import_cv2_np()
    if a.clean and a.out_root.exists():
        shutil.rmtree(a.out_root)
    (a.out_root / "videos").mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_jsonl(a.manifest) if r.get("view_type") == a.view_type][: a.limit]
    base_report = read_report(a.base_report)
    models = PerceptionModels(torch.device("cuda"), a.model_cache, depth_backend="dpt")
    name = f"mf_a{int(round(a.alpha*100)):02d}_h{a.max_history}_{a.base_mode}"
    (a.out_root / name / "videos").mkdir(parents=True, exist_ok=True)
    report = []

    for pos, row in enumerate(rows, 1):
        sid = row["sample_id"]
        base_video = base_report.get(sid)
        conditions = load_images(cv2m, (row.get("frames") or {}).get("condition", []))
        if not base_video or len(conditions) < 2:
            print(f"[{pos}/{len(rows)}] {sid} | skip", flush=True); continue
        bases = load_full_video_frames(cv2m, base_video) if a.preserve_full_video else load_base_frames(cv2m, row, base_video)
        if not bases:
            print(f"[{pos}/{len(rows)}] {sid} | skip(no base)", flush=True); continue
        h, w = bases[0].shape[:2]
        bases = [cv2m.resize(b, (w, h), interpolation=cv2m.INTER_AREA) for b in bases]
        hist = [cv2m.resize(c, (w, h), interpolation=cv2m.INTER_AREA) for c in conditions[-a.max_history:]]
        anchor = hist[-1]
        fillbase = anchor if a.base_mode == "repeat_anchor" else None
        intrinsic = camera_matrix(w, h, a.focal_scale)

        # per-history depth + actor + static masks; pose(h_i -> anchor)
        depths = [models.depth(hf) for hf in hist]
        hactors = models.actor_masks(hist, a.segment_batch)
        statics = [(ha < 0.1).astype(np.float32) for ha in hactors]
        to_anchor = []
        for i, hf in enumerate(hist):
            if i == len(hist) - 1:
                to_anchor.append(IDENTITY)
            else:
                ph, _ = estimate_pose(hf, anchor, depths[i], hactors[i], intrinsic, a.min_matches, a.min_inliers)
                to_anchor.append(ph)  # may be None

        # anchor -> each target pose (smoothed), like the single-frame renderer
        bmasks = models.actor_masks(bases, a.segment_batch)
        raw, quals = [], []
        for b in bases:
            pa, _ = estimate_pose(anchor, b, depths[-1], hactors[-1], intrinsic, a.min_matches, a.min_inliers)
            raw.append(pa); quals.append(0.0 if pa is None else 1.0)
        poses, pose_quals, _ = smooth_pose_sequence(raw, quals, a.pose_smoothing_window, a.filled_pose_scale)

        outputs = []
        prev_conf = None
        for t, base in enumerate(bases):
            pose_at = poses[t]; q = pose_quals[t]
            if pose_at is None or q <= 0:
                outputs.append(fillbase if fillbase is not None else base); prev_conf = None; continue
            fused = (fillbase if fillbase is not None else base).copy()
            cov_acc = np.zeros((h, w), np.float32)
            for i, hf in enumerate(hist):  # oldest -> anchor (anchor last = on top)
                pi = to_anchor[i]
                if pi is None:
                    continue
                pose_it = compose_pose(pi, pose_at)
                rendered_i, cov_i = zbuffer_splat(hf, depths[i], statics[i], pose_it, intrinsic, a.point_stride, a.splat_radius)
                m = cov_i > 0.5
                fused[m] = rendered_i[m]
                cov_acc = np.maximum(cov_acc, cov_i)
            conf = render_confidence(base, fused, cov_acc, bmasks[t], q, a.photo_sigma, prev_conf, a.confidence_ema)
            outputs.append(compose(fillbase if fillbase is not None else base, fused, conf, a.alpha))
            prev_conf = conf

        fps = video_fps(cv2m, base_video) or 30.0
        vp = a.out_root / name / "videos" / f"{sid}.avi"
        write_video(cv2m, outputs, vp, fps, prepend_guard=not a.preserve_full_video)
        report.append({"sample_id": sid, "video": str(vp), "status": "ok", "future_frames": len(outputs)})
        print(f"[{pos}/{len(rows)}] {sid} | ok (hist={len(hist)})", flush=True)

    rp = a.out_root / name / "report.csv"
    with rp.open("w", encoding="utf-8", newline="") as hd:
        wr = csv.DictWriter(hd, fieldnames=["sample_id", "video", "status", "future_frames"]); wr.writeheader(); wr.writerows(report)
    print(f"output={a.out_root/name}")


if __name__ == "__main__":
    main()
