"""
Optical Flow Background Disentanglement Module

Implements background-foreground separation using optical flow estimation
and homography-based background warping. This module:

1. Computes dense optical flow between consecutive history frames (RAFT)
2. Separates ego-motion from object motion using RANSAC homography fitting
3. Warps the background deterministically for future frames
4. Produces a motion mask for foreground regions

By bypassing the generative DiT for static background regions, this module
maximizes PSNR and SSIM scores while the DiT focuses on dynamic foreground.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def compute_optical_flow_raft(
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    model_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute dense optical flow using RAFT.

    Args:
        frame1: [B, C, H, W] - source frame
        frame2: [B, C, H, W] - target frame
        model_path: optional path to RAFT weights

    Returns:
        flow: [B, 2, H, W] - optical flow (u, v)
    """
    try:
        from raft import RAFT as RAFTModel
    except ImportError:
        return compute_optical_flow_classic(frame1, frame2)

    device = frame1.device
    model = RAFTModel(model_path=model_path).to(device)
    model.eval()

    frame1 = (frame1 * 255.0).to(torch.uint8)
    frame2 = (frame2 * 255.0).to(torch.uint8)

    with torch.no_grad():
        flow_low, flow_up = model(frame1, frame2, iters=20, test_mode=True)

    return flow_up


def compute_optical_flow_classic(
    frame1: torch.Tensor,
    frame2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute optical flow using classical Lucas-Kanade / Farneback method.
    Fallback when RAFT is not available.
    """
    import cv2

    B, C, H, W = frame1.shape
    flows = []

    for b in range(B):
        f1 = (frame1[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        f2 = (frame2[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        if len(f1.shape) == 3:
            f1_gray = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
            f2_gray = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
        else:
            f1_gray = f1
            f2_gray = f2

        flow = cv2.calcOpticalFlowFarneback(
            f1_gray, f2_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )

        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).to(frame1.device)
        flows.append(flow_tensor)

    return torch.stack(flows, dim=0)


def estimate_homography_ransac(
    flow: torch.Tensor,
    threshold: float = 3.0,
    max_iters: int = 1000,
) -> torch.Tensor:
    """
    Estimate global homography matrix from optical flow using RANSAC.

    Args:
        flow: [B, 2, H, W] - optical flow field
        threshold: inlier threshold in pixels
        max_iters: maximum RANSAC iterations

    Returns:
        H: [B, 3, 3] - homography matrices
    """
    import cv2

    B, _, H, W = flow.shape
    homographies = []

    for b in range(B):
        flow_b = flow[b].permute(1, 2, 0).cpu().numpy()

        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        pts1 = np.stack([x_coords, y_coords], axis=-1).reshape(-1, 2).astype(np.float32)
        pts2 = pts1 + flow_b.reshape(-1, 2)

        valid_mask = (
            (pts2[:, 0] >= 0) & (pts2[:, 0] < W) &
            (pts2[:, 1] >= 0) & (pts2[:, 1] < H)
        )
        pts1 = pts1[valid_mask]
        pts2 = pts2[valid_mask]

        if len(pts1) < 4:
            homographies.append(np.eye(3))
            continue

        H_mat, inliers = cv2.findHomography(
            pts1, pts2, cv2.RANSAC, threshold, maxIters=max_iters
        )

        if H_mat is None:
            homographies.append(np.eye(3))
        else:
            homographies.append(H_mat)

    return torch.from_numpy(np.stack(homographies)).to(flow.device).float()


def compute_motion_mask_from_warp(
    frame_src: torch.Tensor,
    frame_dst: torch.Tensor,
    homography: torch.Tensor,
    diff_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Compute binary motion mask by warping frame_src to frame_dst's viewpoint
    and thresholding pixel-wise differences.

    This is more reliable than optical-flow-based motion detection because:
      - It amplifies slow motion via the temporal gap between src and dst
      - It directly measures "does this pixel belong to the static background?"
      - Farneback flow is unreliable for small objects at 480p

    Args:
        frame_src: [B, C, H, W] - earlier frame (to be warped)
        frame_dst: [B, C, H, W] - later frame (reference)
        homography: [B, 3, 3] - homography aligning src to dst
        diff_threshold: normalized pixel difference threshold [0, 1]

    Returns:
        mask: [B, 1, H, W] - motion mask (1 = dynamic, 0 = static)
    """
    B, C, H, W = frame_src.shape

    # Build identity grid
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dst_pts = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)
    dst_pts_flat = dst_pts.reshape(-1, 3).T  # [3, H*W]

    masks = []
    for b in range(B):
        H_inv = np.linalg.inv(homography[b].cpu().numpy())

        # Map each dst pixel back to src
        src_pts = H_inv @ dst_pts_flat
        src_pts = src_pts[:2] / src_pts[2:3]  # [2, H*W]

        src_x = 2.0 * src_pts[0].reshape(H, W) / (W - 1) - 1.0
        src_y = 2.0 * src_pts[1].reshape(H, W) / (H - 1) - 1.0

        grid = torch.from_numpy(np.stack([src_x, src_y], axis=-1)).to(frame_src.device).float()
        grid = grid.unsqueeze(0)  # [1, H, W, 2]

        # Warp frame_src to dst viewpoint
        warped_src = F.grid_sample(
            frame_src[b:b+1], grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )  # [1, C, H, W]

        # Pixel-wise difference (normalized in [0, 1] space)
        diff = (warped_src - frame_dst[b:b+1]).abs().mean(dim=1, keepdim=True)  # [1, 1, H, W]
        mask = (diff > diff_threshold).float()

        masks.append(mask.squeeze(0))

    return torch.stack(masks, dim=0)


def warp_background(
    frame: torch.Tensor,
    homography: torch.Tensor,
    output_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Warp background frame using homography for deterministic background synthesis.

    Args:
        frame: [B, C, H, W] - source frame
        homography: [B, 3, 3] - homography matrix
        output_size: optional (H, W) for output

    Returns:
        warped: [B, C, H, W] - warped background
    """
    B, C, H, W = frame.shape
    if output_size is not None:
        H, W = output_size

    grids = []
    for b in range(B):
        H_mat = homography[b].cpu().numpy()
        H_inv = np.linalg.inv(H_mat)

        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        dst_pts = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)
        dst_pts = dst_pts.reshape(-1, 3).T

        src_pts = H_inv @ dst_pts
        src_pts = src_pts[:2] / src_pts[2:3]

        src_x = 2.0 * src_pts[0].reshape(H, W) / (W - 1) - 1.0
        src_y = 2.0 * src_pts[1].reshape(H, W) / (H - 1) - 1.0

        grid = torch.stack([
            torch.from_numpy(src_x).to(frame.device).float(),
            torch.from_numpy(src_y).to(frame.device).float(),
        ], dim=-1)

        grids.append(grid)

    grid = torch.stack(grids, dim=0)
    warped = F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=True)

    return warped


class OpticalFlowDisentanglement(nn.Module):
    """
    Complete optical flow background disentanglement module.

    Pipeline:
    1. Compute dense optical flow between consecutive frames
    2. Estimate global homography via RANSAC
    3. Compute motion mask (dynamic vs static)
    4. Warp background deterministically

    This module enables the SOTA pipeline to bypass generative diffusion
    for static backgrounds, maximizing PSNR/SSIM.
    """

    def __init__(
        self,
        use_raft: bool = False,
        raft_model_path: Optional[str] = None,
        ransac_threshold: float = 1.0,
        motion_threshold: float = 0.05,
        flow_stride: int = 4,
    ):
        super().__init__()
        self.use_raft = use_raft
        self.raft_model_path = raft_model_path
        self.ransac_threshold = ransac_threshold
        self.motion_threshold = motion_threshold
        self.flow_stride = flow_stride

    def forward(
        self,
        history_frames: torch.Tensor,
        num_future_frames: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process history frames to extract background motion model.

        Motion detection uses frame-differencing with homography warp
        (frame[0] vs frame[-1]) instead of optical-flow thresholding.
        This amplifies small motion signals by ~15x and is far more
        reliable for detecting slow pedestrians at 480p resolution.

        Args:
            history_frames: [B, T, C, H, W] - history video frames

        Returns:
            homography: [B, 3, 3] - estimated camera motion
            motion_mask: [B, 1, H, W] - foreground motion mask
            flow: [B, 2, H, W] - average optical flow
        """
        B, T, C, H, W = history_frames.shape

        # Denormalize from [-1.0, 1.0] to [0.0, 1.0] if needed to prevent uint8 overflow in optical flow
        if history_frames.min() < 0.0:
            history_frames = (history_frames * 0.5 + 0.5).clamp(0.0, 1.0)

        # Use stride-based flow pairs for robust homography estimation
        flows = []
        for t in range(0, T - 1, self.flow_stride):
            end = min(t + self.flow_stride, T - 1)
            frame1 = history_frames[:, t]
            frame2 = history_frames[:, end]

            if self.use_raft:
                flow = compute_optical_flow_raft(frame1, frame2, self.raft_model_path)
            else:
                flow = compute_optical_flow_classic(frame1, frame2)

            flows.append(flow)

        if len(flows) == 0:
            flows.append(torch.zeros(B, 2, H, W, device=history_frames.device))

        avg_flow = torch.stack(flows, dim=0).mean(dim=0)

        homography = estimate_homography_ransac(avg_flow, self.ransac_threshold)

        # Use frame-differencing with warp for motion mask (amplifies slow motion ~15x)
        frame_first = history_frames[:, 0]
        frame_last = history_frames[:, -1]
        motion_mask = compute_motion_mask_from_warp(
            frame_first, frame_last, homography,
            diff_threshold=self.motion_threshold,
        )

        return homography, motion_mask, avg_flow

    def warp_future_background(
        self,
        last_history_frame: torch.Tensor,
        homography: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Generate future background frames by cumulative warping to avoid interpolation blur.

        Args:
            last_history_frame: [B, C, H, W] - last observed frame
            homography: [B, 3, 3] - per-frame homography
            num_frames: number of future frames to generate

        Returns:
            background_frames: [B, num_frames, C, H, W]
        """
        B, C, H, W = last_history_frame.shape
        background_frames = []

        # Start with identity homography of shape [B, 3, 3]
        current_homography = torch.eye(3, device=homography.device, dtype=homography.dtype).unsqueeze(0).repeat(B, 1, 1)

        for _ in range(num_frames):
            # Compute cumulative homography: H_cum^(k) = H_cum^(k-1) @ H
            current_homography = torch.bmm(current_homography, homography)
            # Warp the original sharp last_history_frame directly
            warped = warp_background(last_history_frame, current_homography)
            background_frames.append(warped)

        if not background_frames:
            return last_history_frame.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        return torch.stack(background_frames, dim=1)


class Depth3DGSBackgroundPreservationModule(nn.Module):
    """
    3D Gaussian Splatting / 3D Point-Cloud Splatting Background Preservation.
    Replaces 2D Homography warping with a 3D depth-aware camera-motion extrapolation
    and GPU-based painter's algorithm z-buffer forward-splatting.
    """
    def __init__(self, device="cuda", ransac_threshold=1.0, motion_threshold=0.05, flow_stride=4):
        super().__init__()
        self.device = device
        self.ransac_threshold = ransac_threshold
        self.motion_threshold = motion_threshold
        self.flow_stride = flow_stride
        self.depth_model = None
        self.depth_processor = None
        
    def _init_depth_model(self):
        if self.depth_model is None:
            from transformers import DPTForDepthEstimation, DPTImageProcessor
            from pathlib import Path
            cache_dir = Path("/workspace/depth75_solution/model_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            print("  [3DGS BG] Loading DPT depth model on CPU...")
            self.depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas", cache_dir=cache_dir)
            self.depth_model = DPTForDepthEstimation.from_pretrained(
                "Intel/dpt-hybrid-midas", cache_dir=cache_dir, torch_dtype=torch.float32
            ).eval().to("cpu")

    def estimate_depth(self, img_np):
        self._init_depth_model()
        with torch.no_grad():
            inputs = self.depth_processor(images=img_np, return_tensors="pt")
            inputs = {key: value.to("cpu") for key, value in inputs.items()}
            prediction = self.depth_model(**inputs).predicted_depth[:, None]
            prediction = torch.nn.functional.interpolate(
                prediction, size=img_np.shape[:2], mode="bicubic", align_corners=False
            )[0, 0]
            inverse = prediction.float().cpu().numpy()
            low, high = np.percentile(inverse[np.isfinite(inverse)], [2.0, 98.0])
            normalized = np.clip((inverse - low) / max(high - low, 1e-6), 0.0, 1.0)
            return (1.0 / (0.12 + normalized)).astype(np.float32)

    def splat_frame_gpu(self, anchor, depth, static_mask, R, T, intrinsic, radius=1):
        C, H, W = anchor.shape
        device = anchor.device
        
        # Create grid of coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=device),
            torch.arange(0, W, device=device),
            indexing="ij"
        )
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)
        
        # Filter static pixels
        mask_flat = static_mask.reshape(-1)
        keep = mask_flat > 0.5
        grid_x = grid_x[keep]
        grid_y = grid_y[keep]
        
        if len(grid_x) == 0:
            return anchor, torch.zeros(H, W, device=device)
            
        z = depth.reshape(-1)[keep]
        colors = anchor.reshape(C, -1)[:, keep].t() # [M, C]
        
        # Back-project to 3D
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        pts_3d = torch.stack([
            (grid_x - cx) * z / fx,
            (grid_y - cy) * z / fy,
            z
        ], dim=-1) # [M, 3]
        
        # Transform using R, T
        transformed = pts_3d @ R.t() + T.t() # [M, 3]
        
        # Project back to 2D
        z_proj = transformed[:, 2]
        valid = z_proj > 1e-4
        if not valid.any():
            return torch.zeros_like(anchor), torch.zeros(H, W, device=device)
            
        transformed = transformed[valid]
        colors = colors[valid]
        z_proj = z_proj[valid]
        
        u_proj = fx * transformed[:, 0] / transformed[:, 2] + cx
        v_proj = fy * transformed[:, 1] / transformed[:, 2] + cy
        
        u_int = torch.round(u_proj).long()
        v_int = torch.round(v_proj).long()
        
        in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        if not in_bounds.any():
            return torch.zeros_like(anchor), torch.zeros(H, W, device=device)
            
        u_int = u_int[in_bounds]
        v_int = v_int[in_bounds]
        colors = colors[in_bounds]
        z_proj = z_proj[in_bounds]
        
        if radius > 0:
            offsets = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    offsets.append((dx, dy))
            num_offsets = len(offsets)
            
            u_all = u_int.unsqueeze(1).repeat(1, num_offsets)
            v_all = v_int.unsqueeze(1).repeat(1, num_offsets)
            
            for idx, (dx, dy) in enumerate(offsets):
                u_all[:, idx] += dx
                v_all[:, idx] += dy
                
            u_all = u_all.view(-1)
            v_all = v_all.view(-1)
            colors_all = colors.unsqueeze(1).repeat(1, num_offsets, 1).view(-1, C)
            z_proj_all = z_proj.unsqueeze(1).repeat(1, num_offsets).view(-1)
            
            in_bounds_all = (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H)
            u_final = u_all[in_bounds_all]
            v_final = v_all[in_bounds_all]
            colors_final = colors_all[in_bounds_all]
            z_final = z_proj_all[in_bounds_all]
        else:
            u_final = u_int
            v_final = v_int
            colors_final = colors
            z_final = z_proj
            
        if len(u_final) == 0:
            return torch.zeros_like(anchor), torch.zeros(H, W, device=device)
            
        idx_sort = torch.argsort(z_final, descending=True)
        u_sorted = u_final[idx_sort]
        v_sorted = v_final[idx_sort]
        colors_sorted = colors_final[idx_sort]
        
        flat_idx = v_sorted * W + u_sorted
        
        canvas = torch.zeros(C, H * W, device=device)
        canvas.scatter_(1, flat_idx.unsqueeze(0).expand(C, -1), colors_sorted.t())
        rendered = canvas.view(C, H, W)
        
        coverage_flat = torch.zeros(H * W, device=device)
        coverage_flat.scatter_(0, flat_idx, 1.0)
        coverage = coverage_flat.view(H, W)
        
        return rendered, coverage

    def forward(self, history_frames: torch.Tensor, num_future_frames: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        device = history_frames.device
        B, T, C, H, W = history_frames.shape
        
        hist_denorm = history_frames
        if history_frames.min() < 0.0:
            hist_denorm = (history_frames * 0.5 + 0.5).clamp(0.0, 1.0)
            
        flow_module = OpticalFlowDisentanglement(
            use_raft=False,
            ransac_threshold=self.ransac_threshold,
            motion_threshold=self.motion_threshold,
            flow_stride=self.flow_stride,
        ).to(device)
        homography, motion_mask, _ = flow_module(hist_denorm)
        
        last_frame_np = (hist_denorm[0, -1].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        depth_np = self.estimate_depth(last_frame_np)
        depth = torch.from_numpy(depth_np).to(device)
        
        img_prev = (hist_denorm[0, -2].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img_curr = last_frame_np
        
        import cv2
        orb = cv2.ORB_create(nfeatures=2000, fastThreshold=8)
        kp1, des1 = orb.detectAndCompute(img_prev, None)
        kp2, des2 = orb.detectAndCompute(img_curr, None)
        
        R_est, T_est = None, None
        if des1 is not None and des2 is not None and len(kp1) > 20 and len(kp2) > 20:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            pairs = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for pair in pairs:
                if len(pair) == 2:
                    first, second = pair
                    if first.distance < 0.75 * second.distance:
                        good_matches.append(first)
                        
            if len(good_matches) >= 15:
                pts_prev = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                pts_curr = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                
                motion_mask_np = motion_mask[0, 0].cpu().numpy()
                static_matches = []
                for p_prev, p_curr in zip(pts_prev, pts_curr):
                    x, y = int(p_prev[0]), int(p_prev[1])
                    x = min(max(x, 0), W - 1)
                    y = min(max(y, 0), H - 1)
                    if motion_mask_np[y, x] < 0.5:
                        static_matches.append((p_prev, p_curr))
                        
                if len(static_matches) >= 12:
                    image_points = np.float32([m[1] for m in static_matches])
                    pixels = np.float32([m[0] for m in static_matches])
                    
                    px = np.clip(np.rint(pixels[:, 0]).astype(int), 0, W - 1)
                    py = np.clip(np.rint(pixels[:, 1]).astype(int), 0, H - 1)
                    z = depth_np[py, px]
                    
                    focal = 0.9 * max(W, H)
                    intrinsic_np = np.array([[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                    fx, fy, cx, cy = intrinsic_np[0, 0], intrinsic_np[1, 1], intrinsic_np[0, 2], intrinsic_np[1, 2]
                    object_points = np.stack([(pixels[:, 0] - cx) * z / fx, (pixels[:, 1] - cy) * z / fy, z], axis=1)
                    
                    ok, rotation, translation, inliers = cv2.solvePnPRansac(
                        object_points,
                        image_points,
                        intrinsic_np,
                        None,
                        iterationsCount=200,
                        reprojectionError=5.0,
                        confidence=0.99,
                        flags=cv2.SOLVEPNP_EPNP,
                    )
                    if ok and inliers is not None and len(inliers) >= 8:
                        R_est, _ = cv2.Rodrigues(rotation)
                        T_est = translation.reshape(3, 1)
                        
        if R_est is None or T_est is None:
            R_est = np.eye(3)
            T_est = np.zeros((3, 1))
            
        poses_extrapolated = []
        R_cum = np.eye(3)
        T_cum = np.zeros((3, 1))
        for k in range(num_future_frames):
            T_cum = R_est @ T_cum + T_est
            R_cum = R_cum @ R_est
            poses_extrapolated.append((R_cum.copy(), T_cum.copy()))
            
        anchor = hist_denorm[0, -1] # [C, H, W]
        static_mask = (motion_mask[0, 0] < 0.5).float() # [H, W]
        
        focal = 0.9 * max(W, H)
        intrinsic = torch.tensor([[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        
        background_frames = []
        for k in range(num_future_frames):
            R_k, T_k = poses_extrapolated[k]
            R_t = torch.from_numpy(R_k).to(device=device, dtype=torch.float32)
            T_t = torch.from_numpy(T_k).to(device=device, dtype=torch.float32)
            
            rendered, coverage = self.splat_frame_gpu(anchor, depth, static_mask, R_t, T_t, intrinsic, radius=1)
            
            uncovered = (coverage < 0.5).unsqueeze(0).float()
            rendered = rendered * (1.0 - uncovered) + anchor * uncovered
            
            background_frames.append(rendered)
            
        background = torch.stack(background_frames, dim=0).unsqueeze(0) # [1, num_future, C, H, W]
        
        if history_frames.min() < 0.0:
            background = background * 2.0 - 1.0
            
        return background, motion_mask


class BackgroundPreservationModule(nn.Module):
    """
    High-level background preservation module.

    Integrates optical flow disentanglement with deterministic warping
    to produce background frames that bypass the generative DiT entirely.
    Supports either 2D Homography or 3D Gaussian Splatting (3DGS) warping.
    """

    def __init__(
        self,
        use_raft: bool = False,
        raft_model_path: Optional[str] = None,
        ransac_threshold: float = 1.0,
        motion_threshold: float = 0.05,
        flow_stride: int = 4,
        use_3dgs: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.use_3dgs = use_3dgs
        if use_3dgs:
            self.dgs_module = Depth3DGSBackgroundPreservationModule(
                device=device,
                ransac_threshold=ransac_threshold,
                motion_threshold=motion_threshold,
                flow_stride=flow_stride,
            )
        else:
            self.flow_module = OpticalFlowDisentanglement(
                use_raft=use_raft,
                raft_model_path=raft_model_path,
                ransac_threshold=ransac_threshold,
                motion_threshold=motion_threshold,
                flow_stride=flow_stride,
            )

    def forward(
        self,
        history_frames: torch.Tensor,
        num_future_frames: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate preserved background and motion mask.

        Args:
            history_frames: [B, T, C, H, W]
            num_future_frames: number of future frames to predict

        Returns:
            background: [B, num_future, C, H, W] - deterministic background
            motion_mask: [B, 1, H, W] - foreground motion mask
        """
        if self.use_3dgs:
            return self.dgs_module(history_frames, num_future_frames)
            
        homography, motion_mask, _ = self.flow_module(history_frames)

        last_frame = history_frames[:, -1]
        background = self.flow_module.warp_future_background(
            last_frame, homography, num_future_frames
        )

        return background, motion_mask
