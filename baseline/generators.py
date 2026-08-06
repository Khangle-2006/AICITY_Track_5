from __future__ import annotations

from pathlib import Path


def copy_or_write_repeat(cv2, last_frame, output_dir: Path, target_n: int, compression: int) -> int:
    count = 0
    params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    bgr = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
    for i in range(target_n):
        path = output_dir / f"{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(path), bgr, params):
            count += 1
    return count


def compute_flow(cv2, np, prev_rgb, last_rgb, flow_width: int):
    h, w = last_rgb.shape[:2]
    scale = min(1.0, flow_width / float(max(w, 1))) if flow_width > 0 else 1.0
    if scale < 1.0:
        sw = max(2, int(round(w * scale)))
        sh = max(2, int(round(h * scale)))
        prev_small = cv2.resize(prev_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        last_small = cv2.resize(last_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        sw, sh = w, h
        prev_small = prev_rgb
        last_small = last_rgb

    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY)
    last_gray = cv2.cvtColor(last_small, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        last_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )

    if (sw, sh) != (w, h):
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[:, :, 0] *= w / float(sw)
        flow[:, :, 1] *= h / float(sh)
    return flow.astype(np.float32)


def warp_rgb(cv2, np, img, flow, scale: float):
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[:, :, 0] * scale
    map_y = grid_y - flow[:, :, 1] * scale
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def flow_decay_scales(target_n: int, decay: float, max_scale: float) -> list[float]:
    scales = []
    acc = 0.0
    step = 1.0
    for _ in range(target_n):
        acc += step
        scales.append(min(acc, max_scale))
        step *= decay
    return scales


def generate_flow_decay_config(
    cv2,
    np,
    frames,
    output_dir: Path,
    target_n: int,
    args,
    *,
    flow_decay: float,
    max_flow_scale: float,
    last_anchor_decay: float,
    min_last_anchor: float,
) -> tuple[int, dict[str, float]]:
    last = frames[-1]
    prev = frames[-2] if len(frames) >= 2 else frames[-1]
    flow = compute_flow(cv2, np, prev, last, args.flow_width)
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    stats = {
        "mean_motion": float(np.mean(mag)),
        "p90_motion": float(np.percentile(mag, 90)),
        "max_motion": float(np.max(mag)),
    }

    params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
    scales = flow_decay_scales(target_n, decay=flow_decay, max_scale=max_flow_scale)
    count = 0
    for i, scale in enumerate(scales):
        pred = warp_rgb(cv2, np, last, flow, scale)
        anchor = max(min_last_anchor, last_anchor_decay ** (i + 1))
        if anchor > 0:
            pred_f = pred.astype(np.float32)
            last_f = last.astype(np.float32)
            pred = np.clip(anchor * last_f + (1.0 - anchor) * pred_f, 0, 255).astype(np.uint8)
        out_path = output_dir / f"{i}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(out_path), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR), params):
            count += 1
    return count, stats


def generate_flow_decay(cv2, np, frames, output_dir: Path, target_n: int, args) -> tuple[int, dict[str, float]]:
    return generate_flow_decay_config(
        cv2,
        np,
        frames,
        output_dir,
        target_n,
        args,
        flow_decay=args.flow_decay,
        max_flow_scale=args.max_flow_scale,
        last_anchor_decay=args.last_anchor_decay,
        min_last_anchor=args.min_last_anchor,
    )


def generate_flow_light(cv2, np, frames, output_dir: Path, target_n: int, args) -> tuple[int, dict[str, float]]:
    return generate_flow_decay_config(
        cv2,
        np,
        frames,
        output_dir,
        target_n,
        args,
        flow_decay=args.light_flow_decay,
        max_flow_scale=args.light_max_flow_scale,
        last_anchor_decay=args.light_last_anchor_decay,
        min_last_anchor=args.light_min_last_anchor,
    )


def motion_gate(np, p90_motion: float, view_type: str, args) -> float:
    gate = float(1.0 / (1.0 + np.exp(-(p90_motion - args.motion_threshold) / max(args.motion_temperature, 1e-6))))
    if view_type == "wts_overhead_or_fixed_camera":
        gate *= args.motion_overhead_gate_scale
    gate = max(args.motion_min_gate, min(args.motion_max_gate, gate))
    return gate


def generate_motion_adaptive(
    cv2,
    np,
    frames,
    output_dir: Path,
    target_n: int,
    args,
    *,
    view_type: str = "",
) -> tuple[int, dict[str, float]]:
    last = frames[-1]
    prev = frames[-2] if len(frames) >= 2 else frames[-1]
    flow = compute_flow(cv2, np, prev, last, args.flow_width)
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    p90_motion = float(np.percentile(mag, 90))
    gate = motion_gate(np, p90_motion, view_type, args)

    stats = {
        "mean_motion": float(np.mean(mag)),
        "p90_motion": p90_motion,
        "max_motion": float(np.max(mag)),
        "gate": gate,
    }

    params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
    scales = flow_decay_scales(target_n, decay=args.flow_decay, max_scale=args.max_flow_scale)
    count = 0
    for i, scale in enumerate(scales):
        flow_pred = warp_rgb(cv2, np, last, flow, scale)
        pred_f = gate * flow_pred.astype(np.float32) + (1.0 - gate) * last.astype(np.float32)
        pred = np.clip(pred_f, 0, 255).astype(np.uint8)
        out_path = output_dir / f"{i}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(out_path), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR), params):
            count += 1
    return count, stats


def make_region_motion_mask(cv2, np, flow, view_type: str, args):
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    mask = 1.0 / (1.0 + np.exp(-(mag - args.region_motion_tau) / max(args.region_motion_temp, 1e-6)))

    if view_type == "wts_overhead_or_fixed_camera":
        max_alpha = args.region_overhead_max_alpha
    else:
        max_alpha = args.region_max_alpha

    mask = np.clip(mask * max_alpha, 0.0, max_alpha).astype(np.float32)
    if args.region_mask_blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=args.region_mask_blur, sigmaY=args.region_mask_blur)
    mask = np.clip(mask, 0.0, max_alpha).astype(np.float32)
    return mask, mag


def make_actor_guard_mask(cv2, np, frames, flow):
    """Find foreground-ish moving regions where repeated warping tends to smear."""
    last = frames[-1]
    background = np.median(np.stack(frames, axis=0).astype(np.float32), axis=0)
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    color_delta = np.mean(np.abs(last.astype(np.float32) - background), axis=2)
    seed = ((color_delta > 8.0) & (mag > 0.20)) | (color_delta > 24.0)
    mask = seed.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)

    h, w = mask.shape
    filtered = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0.12 * h * w
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20.0 <= area <= max_area:
            cv2.drawContours(filtered, [contour], -1, 255, thickness=-1)

    filtered = cv2.GaussianBlur(filtered.astype(np.float32) / 255.0, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)


def generate_region_flow_blend(
    cv2,
    np,
    frames,
    output_dir: Path,
    target_n: int,
    args,
    *,
    view_type: str = "",
) -> tuple[int, dict[str, float]]:
    """Blend repeat and flow warp per pixel using a soft motion mask.

    Static regions stay close to the last history frame, while high-motion
    regions are allowed to follow flow. This controls the PSNR/AbsDiff tradeoff
    better than one global gate for the whole image.
    """
    last = frames[-1]
    prev = frames[-2] if len(frames) >= 2 else frames[-1]
    flow = compute_flow(cv2, np, prev, last, args.flow_width)
    base_mask, mag = make_region_motion_mask(cv2, np, flow, view_type, args)

    stats = {
        "mean_motion": float(np.mean(mag)),
        "p90_motion": float(np.percentile(mag, 90)),
        "max_motion": float(np.max(mag)),
        "mask_mean": float(np.mean(base_mask)),
        "mask_p90": float(np.percentile(base_mask, 90)),
    }

    if view_type == "wts_overhead_or_fixed_camera":
        max_scale = args.region_overhead_max_flow_scale
    else:
        max_scale = args.region_max_flow_scale

    params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
    scales = flow_decay_scales(target_n, decay=args.region_flow_decay, max_scale=max_scale)
    count = 0
    last_f = last.astype(np.float32)
    for i, scale in enumerate(scales):
        flow_pred = warp_rgb(cv2, np, last, flow, scale).astype(np.float32)
        horizon_gain = min(1.0, args.region_horizon_gain_base + args.region_horizon_gain_step * i)
        mask = np.clip(base_mask * horizon_gain, 0.0, 1.0)[:, :, None]
        pred = np.clip((1.0 - mask) * last_f + mask * flow_pred, 0, 255).astype(np.uint8)
        out_path = output_dir / f"{i}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(out_path), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR), params):
            count += 1
    return count, stats


def generate_region_norm_medium(
    cv2,
    np,
    frames,
    output_dir: Path,
    target_n: int,
    args,
    *,
    view_type: str = "",
    history_gap: int = 1,
) -> tuple[int, dict[str, float]]:
    """Time-normalized regional flow matching the validated motion baseline."""
    last = frames[-1]
    prev = frames[-2] if len(frames) >= 2 else frames[-1]
    flow = compute_flow(cv2, np, prev, last, args.flow_width)
    mask, mag = make_region_motion_mask(cv2, np, flow, view_type, args)
    actor_guard_strength = float(getattr(args, "region_actor_guard_strength", 0.0))
    actor_guard = make_actor_guard_mask(cv2, np, frames, flow) if actor_guard_strength > 0.0 else None

    stats = {
        "mean_motion": float(np.mean(mag)),
        "p90_motion": float(np.percentile(mag, 90)),
        "max_motion": float(np.max(mag)),
        "mask_mean": float(np.mean(mask)),
        "mask_p90": float(np.percentile(mask, 90)),
        "history_gap": float(max(1, history_gap)),
    }
    if actor_guard is not None:
        stats["actor_guard_mean"] = float(np.mean(actor_guard))
        stats["actor_guard_p90"] = float(np.percentile(actor_guard, 90))
        stats["actor_guard_strength"] = actor_guard_strength

    output_dir.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
    last_f = last.astype(np.float32)
    scale = 0.0
    velocity = 1.0 / max(1, history_gap)
    count = 0
    for i in range(target_n):
        scale = min(args.region_norm_max_flow_scale, scale + velocity)
        velocity *= args.region_norm_decay
        flow_pred = warp_rgb(cv2, np, last, flow, scale).astype(np.float32)
        pred = np.clip((1.0 - mask[:, :, None]) * last_f + mask[:, :, None] * flow_pred, 0, 255).astype(np.uint8)
        if actor_guard is not None:
            pred_f = pred.astype(np.float32)
            guard = actor_guard[:, :, None] * actor_guard_strength
            pred = np.clip((1.0 - guard) * pred_f + guard * last_f, 0, 255).astype(np.uint8)
        out_path = output_dir / f"{i}.png"
        if cv2.imwrite(str(out_path), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR), params):
            count += 1
    return count, stats


def generate_actor_layer_small_filled(
    cv2,
    np,
    frames,
    output_dir: Path,
    target_n: int,
    args,
    *,
    history_gap: int = 1,
) -> tuple[int, dict[str, float]]:
    """Move only small foreground-like blobs over a median background.

    This is intended for fixed/overhead WTS views where dense regional flow
    can smear tiny actors but large foreground reflections should be ignored.
    """
    last = frames[-1]
    prev = frames[-2] if len(frames) >= 2 else frames[-1]
    background = np.median(np.stack(frames, axis=0).astype(np.float32), axis=0)
    flow = compute_flow(cv2, np, prev, last, args.flow_width)
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    color_delta = np.mean(np.abs(last.astype(np.float32) - background), axis=2)
    seed = ((mag > args.actor_layer_motion_tau) & (color_delta > args.actor_layer_color_tau)) | (
        color_delta > args.actor_layer_color_high
    )
    mask = seed.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    max_area = float(args.actor_layer_max_area_frac) * mask.shape[0] * mask.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if args.actor_layer_min_area <= area <= max_area:
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

    source_mask = cv2.GaussianBlur(
        filled.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=args.actor_layer_mask_blur,
        sigmaY=args.actor_layer_mask_blur,
    )
    source_mask = np.clip(source_mask, 0.0, 1.0).astype(np.float32)
    clean_background = (
        (1.0 - source_mask[:, :, None]) * last.astype(np.float32)
        + source_mask[:, :, None] * background
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
    mask_rgb = np.repeat((source_mask[:, :, None] * 255.0).astype(np.uint8), 3, axis=2)
    scale = 0.0
    velocity = 1.0 / max(1, history_gap)
    count = 0
    for index in range(target_n):
        scale = min(args.actor_layer_max_flow_scale, scale + velocity)
        velocity *= args.actor_layer_decay
        warped_actor = warp_rgb(cv2, np, last, flow, scale).astype(np.float32)
        destination_mask = warp_rgb(cv2, np, mask_rgb, flow, scale)[:, :, :1].astype(np.float32) / 255.0
        destination_mask = np.clip(destination_mask, 0.0, 1.0)
        pred = (1.0 - destination_mask) * clean_background + destination_mask * warped_actor
        out_path = output_dir / f"{index}.png"
        if cv2.imwrite(str(out_path), cv2.cvtColor(np.clip(pred, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR), params):
            count += 1

    stats = {
        "mean_motion": float(np.mean(mag)),
        "p90_motion": float(np.percentile(mag, 90)),
        "max_motion": float(np.max(mag)),
        "mask_mean": float(np.mean(source_mask)),
        "mask_p90": float(np.percentile(source_mask, 90)),
        "history_gap": float(max(1, history_gap)),
    }
    return count, stats
