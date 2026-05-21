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


def compute_motion_mask(
    flow: torch.Tensor,
    homography: torch.Tensor,
    threshold: float = 5.0,
) -> torch.Tensor:
    """
    Compute binary motion mask by finding flow outliers from homography.

    Pixels where flow deviates significantly from the homography prediction
    are classified as dynamic foreground.

    Args:
        flow: [B, 2, H, W] - optical flow
        homography: [B, 3, 3] - homography matrix
        threshold: deviation threshold for motion classification

    Returns:
        mask: [B, 1, H, W] - motion mask (1 = dynamic, 0 = static)
    """
    B, _, H, W = flow.shape
    masks = []

    for b in range(B):
        H_mat = homography[b].cpu().numpy()

        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        pts = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)
        pts = pts.reshape(-1, 3).T

        pts_warped = H_mat @ pts
        pts_warped = pts_warped[:2] / pts_warped[2:3]

        pts_warped = pts_warped.T.reshape(H, W, 2)
        pts_orig = np.stack([x_coords, y_coords], axis=-1)

        flow_homography = pts_warped - pts_orig
        flow_actual = flow[b].permute(1, 2, 0).cpu().numpy()

        deviation = np.linalg.norm(flow_actual - flow_homography, axis=-1)
        mask = (deviation > threshold).astype(np.float32)

        masks.append(torch.from_numpy(mask).to(flow.device))

    return torch.stack(masks, dim=0).unsqueeze(1)


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
    warped = F.grid_sample(frame, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

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
        ransac_threshold: float = 3.0,
        motion_threshold: float = 5.0,
    ):
        super().__init__()
        self.use_raft = use_raft
        self.raft_model_path = raft_model_path
        self.ransac_threshold = ransac_threshold
        self.motion_threshold = motion_threshold

    def forward(
        self,
        history_frames: torch.Tensor,
        num_future_frames: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process history frames to extract background motion model.

        Args:
            history_frames: [B, T, C, H, W] - history video frames

        Returns:
            homography: [B, 3, 3] - estimated camera motion
            motion_mask: [B, 1, H, W] - foreground motion mask
            flow: [B, 2, H, W] - average optical flow
        """
        B, T, C, H, W = history_frames.shape

        flows = []
        for t in range(T - 1):
            frame1 = history_frames[:, t]
            frame2 = history_frames[:, t + 1]

            if self.use_raft:
                flow = compute_optical_flow_raft(frame1, frame2, self.raft_model_path)
            else:
                flow = compute_optical_flow_classic(frame1, frame2)

            flows.append(flow)

        avg_flow = torch.stack(flows, dim=0).mean(dim=0)

        homography = estimate_homography_ransac(avg_flow, self.ransac_threshold)
        motion_mask = compute_motion_mask(avg_flow, homography, self.motion_threshold)

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

        return torch.stack(background_frames, dim=1)


class BackgroundPreservationModule(nn.Module):
    """
    High-level background preservation module.

    Integrates optical flow disentanglement with deterministic warping
    to produce background frames that bypass the generative DiT entirely.
    """

    def __init__(
        self,
        use_raft: bool = False,
        raft_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.flow_module = OpticalFlowDisentanglement(
            use_raft=use_raft,
            raft_model_path=raft_model_path,
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
        homography, motion_mask, _ = self.flow_module(history_frames)

        last_frame = history_frames[:, -1]
        background = self.flow_module.warp_future_background(
            last_frame, homography, num_future_frames
        )

        return background, motion_mask
