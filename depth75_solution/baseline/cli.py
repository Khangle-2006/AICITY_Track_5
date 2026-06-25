from __future__ import annotations

import argparse

from .runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--out", default="/workspace/experiments/baseline_e2e")
    parser.add_argument(
        "--method",
        choices=[
            "repeat",
            "flow_decay",
            "flow_light",
            "view_hybrid",
            "motion_adaptive",
            "region_flow_blend",
            "region_norm_medium",
            "actor_layer_small_filled",
        ],
        default="flow_decay",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process first N sorted test rows. 0 means all.")
    parser.add_argument("--view_type", default="", help="Optionally process only one camera view.")
    parser.add_argument("--phase", default="", help="Optionally process only one event phase.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no_clean", action="store_false", dest="clean")
    parser.add_argument("--make_montage", action="store_true")

    parser.add_argument("--flow_width", type=int, default=426)
    parser.add_argument("--flow_decay", type=float, default=0.92)
    parser.add_argument("--max_flow_scale", type=float, default=12.0)
    parser.add_argument("--last_anchor_decay", type=float, default=0.97)
    parser.add_argument("--min_last_anchor", type=float, default=0.15)
    parser.add_argument("--light_flow_decay", type=float, default=0.80)
    parser.add_argument("--light_max_flow_scale", type=float, default=4.0)
    parser.add_argument("--light_last_anchor_decay", type=float, default=0.985)
    parser.add_argument("--light_min_last_anchor", type=float, default=0.45)
    parser.add_argument("--motion_threshold", type=float, default=2.0)
    parser.add_argument("--motion_temperature", type=float, default=1.0)
    parser.add_argument("--motion_min_gate", type=float, default=0.05)
    parser.add_argument("--motion_max_gate", type=float, default=0.95)
    parser.add_argument("--motion_overhead_gate_scale", type=float, default=0.65)
    parser.add_argument("--region_motion_tau", type=float, default=1.5)
    parser.add_argument("--region_motion_temp", type=float, default=0.75)
    parser.add_argument("--region_max_alpha", type=float, default=0.85)
    parser.add_argument("--region_overhead_max_alpha", type=float, default=0.45)
    parser.add_argument("--region_mask_blur", type=float, default=3.0)
    parser.add_argument("--region_flow_decay", type=float, default=0.92)
    parser.add_argument("--region_max_flow_scale", type=float, default=10.0)
    parser.add_argument("--region_overhead_max_flow_scale", type=float, default=4.0)
    parser.add_argument("--region_horizon_gain_base", type=float, default=0.85)
    parser.add_argument("--region_horizon_gain_step", type=float, default=0.02)
    parser.add_argument("--region_norm_decay", type=float, default=0.96)
    parser.add_argument("--region_norm_max_flow_scale", type=float, default=6.0)
    parser.add_argument("--actor_layer_decay", type=float, default=0.985)
    parser.add_argument("--actor_layer_max_flow_scale", type=float, default=12.0)
    parser.add_argument("--actor_layer_motion_tau", type=float, default=0.35)
    parser.add_argument("--actor_layer_color_tau", type=float, default=7.0)
    parser.add_argument("--actor_layer_color_high", type=float, default=22.0)
    parser.add_argument("--actor_layer_min_area", type=float, default=20.0)
    parser.add_argument("--actor_layer_max_area_frac", type=float, default=0.025)
    parser.add_argument("--actor_layer_mask_blur", type=float, default=1.0)
    parser.add_argument(
        "--region_actor_guard_strength",
        type=float,
        default=0.0,
        help="Blend region_norm_medium predictions back toward the last frame on moving foreground-like regions.",
    )
    parser.add_argument("--png_compression", type=int, default=3)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
