from __future__ import annotations

from dataclasses import dataclass

from .generators import compute_flow, flow_decay_scales, make_region_motion_mask, warp_rgb


VIEW_TO_INDEX = {
    "bdd": 0,
    "wts_overhead_or_fixed_camera": 1,
    "wts_vehicle_or_ip_camera": 2,
}


@dataclass
class RefinerConfig:
    flow_width: int = 426
    flow_decay: float = 0.92
    max_flow_scale: float = 12.0
    light_flow_decay: float = 0.80
    light_max_flow_scale: float = 4.0
    region_motion_tau: float = 1.5
    region_motion_temp: float = 0.75
    region_max_alpha: float = 0.85
    region_overhead_max_alpha: float = 0.45
    region_mask_blur: float = 3.0
    time_normalized_features: bool = False
    include_context_features: bool = False
    max_context_horizon: int = 120
    max_context_history_gap: int = 30


def build_feature_config(args) -> RefinerConfig:
    return RefinerConfig(
        flow_width=args.flow_width,
        flow_decay=args.flow_decay,
        max_flow_scale=args.max_flow_scale,
        light_flow_decay=args.light_flow_decay,
        light_max_flow_scale=args.light_max_flow_scale,
        region_motion_tau=args.region_motion_tau,
        region_motion_temp=args.region_motion_temp,
        region_max_alpha=args.region_max_alpha,
        region_overhead_max_alpha=args.region_overhead_max_alpha,
        region_mask_blur=args.region_mask_blur,
        time_normalized_features=getattr(args, "time_normalized_features", False),
        include_context_features=getattr(args, "include_context_features", False),
        max_context_horizon=getattr(args, "max_context_horizon", 120),
        max_context_history_gap=getattr(args, "max_context_history_gap", 30),
    )


def refiner_in_channels(cfg: RefinerConfig) -> int:
    return 20 if cfg.include_context_features else 13


def view_one_hot(view_type: str) -> list[float]:
    out = [0.0, 0.0, 0.0]
    out[VIEW_TO_INDEX.get(view_type, 0)] = 1.0
    return out


def normalized_flow_scale(horizon: int, history_gap: int, decay: float, max_scale: float) -> float:
    scale = 0.0
    velocity = 1.0 / max(1, history_gap)
    for _ in range(max(1, horizon)):
        scale = min(max_scale, scale + velocity)
        velocity *= decay
    return scale


def base_scale_for_view(view_type: str, horizon: int, cfg: RefinerConfig, history_gap: int = 1) -> float:
    if cfg.time_normalized_features:
        decay = cfg.light_flow_decay if view_type == "wts_overhead_or_fixed_camera" else cfg.flow_decay
        max_scale = (
            cfg.light_max_flow_scale
            if view_type == "wts_overhead_or_fixed_camera"
            else cfg.max_flow_scale
        )
        return normalized_flow_scale(horizon, history_gap, decay, max_scale)
    if view_type == "wts_overhead_or_fixed_camera":
        return flow_decay_scales(horizon, cfg.light_flow_decay, cfg.light_max_flow_scale)[-1]
    return flow_decay_scales(horizon, cfg.flow_decay, cfg.max_flow_scale)[-1]


def make_refiner_arrays(
    cv2,
    np,
    frames,
    target,
    view_type: str,
    horizon: int,
    cfg: RefinerConfig,
    history_gap: int = 1,
    phase: str = "",
):
    """Return input feature tensor arrays and target for the residual refiner.

    Features are image-space, normalized later by the dataset:
      last RGB, flow-warp base RGB, absolute diff RGB, motion mask, view one-hot maps.
    """
    last = frames[-1]
    prev = frames[-2] if len(frames) >= 2 else frames[-1]
    flow = compute_flow(cv2, np, prev, last, cfg.flow_width)
    scale = base_scale_for_view(view_type, max(1, horizon), cfg, history_gap)
    base = warp_rgb(cv2, np, last, flow, scale)

    # Reuse region mask only as an input hint, not as a hard output blend.
    mask_args = type(
        "MaskArgs",
        (),
        {
            "region_motion_tau": cfg.region_motion_tau,
            "region_motion_temp": cfg.region_motion_temp,
            "region_max_alpha": cfg.region_max_alpha,
            "region_overhead_max_alpha": cfg.region_overhead_max_alpha,
            "region_mask_blur": cfg.region_mask_blur,
        },
    )()
    mask, _ = make_region_motion_mask(cv2, np, flow, view_type, mask_args)
    diff = np.abs(base.astype(np.float32) - last.astype(np.float32)).astype(np.uint8)

    h, w = last.shape[:2]
    view = np.zeros((h, w, 3), dtype=np.float32)
    for i, value in enumerate(view_one_hot(view_type)):
        view[:, :, i] = value * 255.0

    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)[:, :, None]
    feature_parts = [last, base, diff, mask_u8, view.astype(np.uint8)]
    if cfg.include_context_features:
        horizon_value = min(1.0, max(0.0, horizon / float(max(1, cfg.max_context_horizon))))
        gap_value = min(1.0, max(0.0, history_gap / float(max(1, cfg.max_context_history_gap))))
        context = np.zeros((h, w, 7), dtype=np.uint8)
        context[:, :, 0] = int(round(horizon_value * 255.0))
        context[:, :, 1] = int(round(gap_value * 255.0))
        try:
            phase_index = int(phase)
        except (TypeError, ValueError):
            phase_index = -1
        if 0 <= phase_index < 5:
            context[:, :, 2 + phase_index] = 255
        feature_parts.append(context)
    features = np.concatenate(feature_parts, axis=2)
    return features, target, base
