"""
Alpha Blending and Foreground-Background Compositing

Implements adaptive foreground-background alpha blending for the SOTA pipeline.
The DiT outputs an additional alpha mask channel that defines the spatial
footprint of dynamic action and disoccluded areas.

Final output = alpha * foreground + (1 - alpha) * background

Depth estimation is used to inform the alpha mask for clean occlusion handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DepthAwareAlphaMask(nn.Module):
    """
    Refines alpha mask using depth estimation for clean occlusion handling.

    Uses depth maps to ensure proper depth ordering when compositing
    generated foreground over warped background.
    """

    def __init__(
        self,
        depth_threshold: float = 0.3,
        smoothing_iterations: int = 3,
    ):
        super().__init__()
        self.depth_threshold = depth_threshold
        self.smoothing_iterations = smoothing_iterations

    def forward(
        self,
        alpha_raw: torch.Tensor,
        depth_foreground: torch.Tensor,
        depth_background: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine alpha mask using depth information.

        Args:
            alpha_raw: [B, 1, T, H, W] - raw alpha from DiT
            depth_foreground: [B, 1, T, H, W] - depth of generated foreground
            depth_background: [B, 1, T, H, W] - depth of warped background

        Returns:
            alpha_refined: [B, 1, T, H, W] - depth-aware alpha mask
        """
        alpha = alpha_raw.clone()

        depth_diff = depth_background - depth_foreground
        occlusion_mask = (depth_diff > self.depth_threshold).float()

        alpha = alpha * occlusion_mask + alpha * (1 - occlusion_mask) * 0.5

        for _ in range(self.smoothing_iterations):
            kernel = torch.ones(1, 1, 3, 3, device=alpha.device) / 9.0
            if alpha.dim() == 5:
                B, C, T, H, W = alpha.shape
                alpha = alpha.reshape(B * C * T, 1, H, W)
                alpha = F.conv2d(alpha, kernel, padding=1)
                alpha = alpha.reshape(B, C, T, H, W)
            else:
                alpha = F.conv2d(alpha, kernel, padding=1)

        return alpha


class AdaptiveAlphaBlending(nn.Module):
    """
    Adaptive foreground-background alpha blending module.

    Composites generated foreground (from DiT) with deterministic
    background (from homography warping) using a learned/predicted alpha mask.

    Formula: output = alpha * foreground + (1 - alpha) * background

    The alpha mask is either:
    1. Predicted by the DiT as an additional output channel
    2. Derived from the motion mask from optical flow
    3. Combined from both sources
    """

    def __init__(
        self,
        use_depth_aware: bool = True,
        depth_threshold: float = 0.3,
        motion_mask_weight: float = 0.5,
    ):
        super().__init__()
        self.use_depth_aware = use_depth_aware
        self.motion_mask_weight = motion_mask_weight

        if use_depth_aware:
            self.depth_refinement = DepthAwareAlphaMask(
                depth_threshold=depth_threshold,
            )

    def forward(
        self,
        foreground: torch.Tensor,
        background: torch.Tensor,
        alpha_mask: torch.Tensor,
        motion_mask: Optional[torch.Tensor] = None,
        depth_foreground: Optional[torch.Tensor] = None,
        depth_background: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Composite foreground and background using alpha blending.

        Args:
            foreground: [B, C, T, H, W] - generated foreground from DiT
            background: [B, C, T, H, W] - deterministic background from warping
            alpha_mask: [B, 1, T, H, W] - alpha mask from DiT
            motion_mask: [B, 1, H, W] - optional motion mask from optical flow
            depth_foreground: optional depth for foreground
            depth_background: optional depth for background

        Returns:
            output: [B, C, T, H, W] - composited output
        """
        if motion_mask is not None:
            if motion_mask.dim() == 4:
                motion_mask = motion_mask.unsqueeze(2)  # [B, 1, 1, H, W]

            T_target = alpha_mask.shape[2]
            if motion_mask.shape[2] != T_target:
                motion_mask = motion_mask.expand(-1, -1, T_target, -1, -1)

            B_m, C_m, T_m, H_m, W_m = motion_mask.shape
            
            # Reshape to 2D for spatial morphological dilation and smoothing
            motion_mask_2d = motion_mask.permute(0, 2, 1, 3, 4).reshape(B_m * T_m, C_m, H_m, W_m)
            
            # Dilation with kernel_size=5 (padding=2) to expand foreground coverage
            dilated = F.max_pool2d(motion_mask_2d, kernel_size=5, stride=1, padding=2)
            
            # Smoothing with box filter to feather edges
            kernel = torch.ones(1, 1, 5, 5, device=motion_mask.device) / 25.0
            feathered = F.conv2d(dilated, kernel, padding=2)
            
            # Reshape back to 5D
            alpha = feathered.reshape(B_m, T_m, C_m, H_m, W_m).permute(0, 2, 1, 3, 4)
        else:
            alpha = torch.sigmoid(alpha_mask)

        if self.use_depth_aware and depth_foreground is not None and depth_background is not None:
            alpha = self.depth_refinement(alpha, depth_foreground, depth_background)

        alpha = torch.clamp(alpha, 0.0, 1.0)

        # Resize alpha to match foreground/background spatial dimensions if they differ
        if alpha.shape[-2:] != foreground.shape[-2:]:
            B_a, _, T_a, H_a, W_a = alpha.shape
            H_tgt, W_tgt = foreground.shape[-2:]
            alpha = F.interpolate(
                alpha.reshape(B_a * T_a, 1, H_a, W_a),
                size=(H_tgt, W_tgt),
                mode="bilinear",
                align_corners=False,
            ).reshape(B_a, 1, T_a, H_tgt, W_tgt)

        output = alpha * foreground + (1 - alpha) * background

        return output

    def blend_single_frame(
        self,
        foreground: torch.Tensor,
        background: torch.Tensor,
        alpha_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend single frame (no temporal dimension).

        Args:
            foreground: [B, C, H, W]
            background: [B, C, H, W]
            alpha_mask: [B, 1, H, W]

        Returns:
            output: [B, C, H, W]
        """
        alpha = torch.sigmoid(alpha_mask)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        return alpha * foreground + (1 - alpha) * background


class DisocclusionHandler(nn.Module):
    """
    Handles disoccluded regions - areas of background newly revealed
    behind a moving object.

    Uses inpainting-style generation to fill these regions smoothly.
    """

    def __init__(self, inpaint_radius: int = 5):
        super().__init__()
        self.inpaint_radius = inpaint_radius

    def forward(
        self,
        foreground: torch.Tensor,
        background: torch.Tensor,
        disocclusion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Handle disoccluded regions by blending with inpainted background.

        Args:
            foreground: [B, C, T, H, W]
            background: [B, C, T, H, W]
            disocclusion_mask: [B, 1, T, H, W] - regions that were occluded

        Returns:
            output: [B, C, T, H, W]
        """
        if disocclusion_mask is None:
            return foreground

        disocclusion_mask = torch.clamp(disocclusion_mask, 0.0, 1.0)

        for _ in range(self.inpaint_radius):
            if disocclusion_mask.dim() == 5:
                B, C, T, H, W = disocclusion_mask.shape
                dm = disocclusion_mask.reshape(B * C * T, 1, H, W)
            else:
                dm = disocclusion_mask

            kernel = torch.ones(1, 1, 3, 3, device=dm.device) / 9.0
            dm = F.conv2d(dm, kernel, padding=1)

            if disocclusion_mask.dim() == 5:
                dm = dm.reshape(B, C, T, H, W)
                disocclusion_mask = dm

        blended = foreground * (1 - disocclusion_mask) + background * disocclusion_mask

        return blended


class ForegroundBackgroundCompositor(nn.Module):
    """
    Complete foreground-background compositing pipeline.

    Integrates:
    1. Alpha blending with depth awareness
    2. Disocclusion handling
    3. Temporal smoothing for flicker reduction
    """

    def __init__(
        self,
        use_depth_aware: bool = True,
        use_disocclusion: bool = True,
        temporal_smoothing: bool = False,
        motion_mask_weight: float = 0.5,
    ):
        super().__init__()
        self.alpha_blending = AdaptiveAlphaBlending(
            use_depth_aware=use_depth_aware,
            motion_mask_weight=motion_mask_weight,
        )
        self.disocclusion = DisocclusionHandler() if use_disocclusion else None
        self.temporal_smoothing = temporal_smoothing

    def forward(
        self,
        foreground: torch.Tensor,
        background: torch.Tensor,
        alpha_mask: torch.Tensor,
        motion_mask: Optional[torch.Tensor] = None,
        disocclusion_mask: Optional[torch.Tensor] = None,
        depth_foreground: Optional[torch.Tensor] = None,
        depth_background: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full compositing pipeline.

        Args:
            foreground: [B, C, T, H, W] - generated foreground
            background: [B, C, T, H, W] - deterministic background
            alpha_mask: [B, 1, T, H, W] - alpha from DiT
            motion_mask: [B, 1, H, W] - motion from optical flow
            disocclusion_mask: [B, 1, T, H, W] - disocclusion regions
            depth_foreground: optional depth maps
            depth_background: optional depth maps

        Returns:
            output: [B, C, T, H, W] - final composited video
        """
        output = self.alpha_blending(
            foreground, background, alpha_mask, motion_mask,
            depth_foreground, depth_background,
        )

        if self.disocclusion is not None and disocclusion_mask is not None:
            output = self.disocclusion(output, background, disocclusion_mask)

        if self.temporal_smoothing and output.dim() == 5:
            output = self._temporal_smooth(output)

        return output

    def _temporal_smooth(self, x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Apply temporal smoothing to reduce flicker."""
        B, C, T, H, W = x.shape
        if T < kernel_size:
            return x

        weights = torch.ones(kernel_size, device=x.device) / kernel_size
        weights = weights.view(1, 1, kernel_size, 1, 1).expand(C, 1, kernel_size, 1, 1)

        x_padded = F.pad(x, (0, 0, 0, 0, kernel_size // 2, kernel_size // 2), mode="replicate")
        smoothed = F.conv3d(x_padded, weights, groups=C)

        return smoothed
