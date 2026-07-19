"""Apply validated actor-flow deghosting to BDD test PNG folders."""

from __future__ import annotations

import argparse
import csv
import os as _os
import shutil
from pathlib import Path

_T = _os.environ.get("CPU_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, _T)

import cv2 as _cv2
import numpy as np
import torch

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output
from wan_ticft.evaluation.materialize_actor_flow_deghost import (
    color_match_paste_region,
    component_validated_alpha,
    dense_flow_base_to_anchor,
    dynamic_anchor_mask,
    erase_nearby_base_ghosts,
    warp_from_anchor,
    warp_valid_mask,
)
from wan_ticft.evaluation.materialize_depth_layered_renderer import PerceptionModels

_cv2.setNumThreads(int(_T))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--flow_scale", type=float, default=0.5)
    parser.add_argument("--actor_strength", type=float, default=0.65)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--mask_blur", type=float, default=3.5)
    parser.add_argument("--segment_batch", type=int, default=4)
    parser.add_argument("--semantic_anchor_mask", action="store_true")
    parser.add_argument("--dynamic_residual_thresh", type=float, default=2.8)
    parser.add_argument("--dynamic_min_area", type=int, default=80)
    parser.add_argument("--dynamic_max_area_frac", type=float, default=0.05)
    parser.add_argument("--dynamic_max_bbox_height_frac", type=float, default=0.35)
    parser.add_argument("--dynamic_exclude_bottom_frac", type=float, default=0.0)
    parser.add_argument("--component_min_area", type=int, default=80)
    parser.add_argument("--component_max_area_frac", type=float, default=0.035)
    parser.add_argument("--component_max_bbox_height_frac", type=float, default=0.30)
    parser.add_argument("--component_exclude_bottom_frac", type=float, default=0.02)
    parser.add_argument("--component_boundary", type=int, default=15)
    parser.add_argument("--component_max_boundary_diff", type=float, default=28.0)
    parser.add_argument("--component_min_valid_frac", type=float, default=0.92)
    parser.add_argument("--component_color_match", action="store_true")
    parser.add_argument("--color_match_boundary", type=int, default=19)
    parser.add_argument("--color_match_max_shift", type=float, default=18.0)
    parser.add_argument("--erase_base_ghosts", action="store_true")
    parser.add_argument("--erase_mask_thresh", type=float, default=0.35)
    parser.add_argument("--erase_alpha_keep_thresh", type=float, default=0.06)
    parser.add_argument("--erase_near_radius", type=int, default=95)
    parser.add_argument("--erase_min_area", type=int, default=80)
    parser.add_argument("--erase_max_area_frac", type=float, default=0.02)
    parser.add_argument("--erase_inpaint_radius", type=float, default=3.0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2m, _ = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = PerceptionModels(torch.device("cuda"), args.model_cache, depth_backend="dpt")
    rows = [row for row in read_csv(args.test_csv) if row.get("view_type") == args.view_type]
    report = []

    for pos, row in enumerate(rows, start=1):
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        condition_paths = numeric_pngs(Path(row["input_dir"]))
        base_paths = numeric_pngs(args.base_dir / output_name)
        conditions = [image for path in condition_paths if (image := read_rgb(cv2m, path)) is not None]
        bases = [image for path in base_paths if (image := read_rgb(cv2m, path)) is not None]
        if len(conditions) < 1 or len(bases) != target_n:
            raise RuntimeError(f"{output_name}: conditions={len(conditions)} bases={len(bases)} target={target_n}")

        h, w = bases[0].shape[:2]
        conditions = [cv2m.resize(image, (w, h), interpolation=cv2m.INTER_AREA) for image in conditions]
        bases = [cv2m.resize(image, (w, h), interpolation=cv2m.INTER_AREA) for image in bases]
        anchor = conditions[-1]
        extra_history = [conditions[-2]] if len(conditions) >= 2 else []
        masks = models.actor_masks([*extra_history, anchor, *bases], args.segment_batch)
        if extra_history:
            previous_mask, anchor_mask, base_masks = masks[0], masks[1], masks[2:]
            if not args.semantic_anchor_mask:
                anchor_mask = dynamic_anchor_mask(
                    cv2m,
                    extra_history[0],
                    anchor,
                    previous_mask,
                    anchor_mask,
                    args.flow_scale,
                    args.dynamic_residual_thresh,
                    args.dynamic_min_area,
                    args.dynamic_max_area_frac,
                    args.dynamic_max_bbox_height_frac,
                    args.dynamic_exclude_bottom_frac,
                )
        else:
            anchor_mask, base_masks = masks[0], masks[1:]

        sample_out = args.out_dir / output_name
        if sample_out.exists():
            shutil.rmtree(sample_out)
        sample_out.mkdir(parents=True)
        accepted_components = 0
        total_components = 0
        alpha_means = []
        erase_means = []
        for frame_index, base in enumerate(bases):
            flow_ba = dense_flow_base_to_anchor(cv2m, base, anchor, args.flow_scale)
            warped_anchor = warp_from_anchor(cv2m, anchor, flow_ba, border_value=(0, 0, 0))
            warped_mask = warp_from_anchor(cv2m, anchor_mask.astype(np.float32), flow_ba, border_value=0)
            warped_valid = warp_valid_mask(cv2m, anchor_mask.shape[:2], flow_ba)
            alpha, diagnostics = component_validated_alpha(
                cv2m,
                np,
                base,
                warped_anchor,
                warped_mask,
                warped_valid,
                args.component_min_area,
                args.component_max_area_frac,
                args.component_max_bbox_height_frac,
                args.component_exclude_bottom_frac,
                args.component_boundary,
                args.component_max_boundary_diff,
                args.component_min_valid_frac,
                args.mask_dilate,
                args.mask_blur,
            )
            base_eff = base
            erase_fraction = 0.0
            if args.erase_base_ghosts:
                base_eff, erase_fraction = erase_nearby_base_ghosts(
                    cv2m,
                    np,
                    base,
                    base_masks[frame_index],
                    alpha,
                    args.erase_mask_thresh,
                    args.erase_alpha_keep_thresh,
                    args.erase_near_radius,
                    args.erase_min_area,
                    args.erase_max_area_frac,
                    args.erase_inpaint_radius,
                )
            if args.component_color_match:
                warped_anchor = color_match_paste_region(
                    cv2m,
                    np,
                    base_eff,
                    warped_anchor,
                    alpha,
                    args.color_match_boundary,
                    args.color_match_max_shift,
                )
            total_components += len(diagnostics)
            accepted_components += sum(1 for item in diagnostics if item["accepted"] > 0.5)
            alpha_means.append(float(np.mean(alpha)))
            erase_means.append(erase_fraction)
            alpha = (alpha * args.actor_strength)[..., None]
            out = np.clip(
                base_eff.astype(np.float32) * (1.0 - alpha) + warped_anchor.astype(np.float32) * alpha,
                0,
                255,
            ).astype(np.uint8)
            if not write_rgb(cv2m, sample_out / f"{frame_index}.png", out):
                raise RuntimeError(f"cannot write {sample_out / f'{frame_index}.png'}")

        verified = verify_sample_output(sample_out, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verification failed: {verified.notes}")
        report.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "view_type": row["view_type"],
                "target_N": target_n,
                "frames_found": verified.frames_found,
                "status": "ok",
                "accepted_components": accepted_components,
                "total_components": total_components,
                "alpha_mean": f"{float(np.mean(alpha_means)):.6f}" if alpha_means else "0",
                "erase_mean": f"{float(np.mean(erase_means)):.6f}" if erase_means else "0",
            }
        )
        print(
            f"[{pos}/{len(rows)}] {output_name} | ok "
            f"components={accepted_components}/{total_components} "
            f"alpha={float(np.mean(alpha_means)) if alpha_means else 0:.4f} "
            f"erase={float(np.mean(erase_means)) if erase_means else 0:.4f}",
            flush=True,
        )

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
