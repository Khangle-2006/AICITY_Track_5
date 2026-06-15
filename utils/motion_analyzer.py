"""
src/models/motion_analyzer.py
===============================
History Frame Motion Analyzer (Refactored)

Extracts motion features from variable-K history frames to:
  1. Measure physical trajectory vectors.
  2. Support dynamic motion thresholding and Bounding Box region analysis.
  3. Export structured JSON/dict to the PromptBuilder for LLM semantic fusion.

Key Fixes Implemented:
  - Coordinate Rigor: Standardized Cartesian mapping (90° = UP).
  - RAFT Batching & Padding: Full GPU batching + F.pad to prevent 
    divisibility-by-8 tensor shape mismatch crashes.
  - The Averaging Curse: Eliminated static background dilution by applying 
    dynamic magnitude thresholding + bounding box masking.
  - Semantic Fusion: Removed global string concatenation to avoid 
    hallucinations in Wan2.2 Text Encoder.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# MOTION VOCABULARY (for LLM Semantic Routing)
# ═══════════════════════════════════════════════════════════════════

SPEED_VOCAB = [
    (0.0, 2.0,    "stationary"),
    (2.0, 8.0,    "slowly moving"),
    (8.0, 20.0,   "moving at moderate speed"),
    (20.0, 40.0,  "moving quickly"),
    (40.0, 999.0, "moving rapidly"),
]

# Cartesian mapping: 0=Right, 90=Up, 180=Left, 270=Down
DIRECTION_VOCAB = [
    (337.5, 22.5,   "toward the right"),
    (22.5, 67.5,    "toward the upper-right"),
    (67.5, 112.5,   "toward the top of frame"),
    (112.5, 157.5,  "toward the upper-left"),
    (157.5, 202.5,  "toward the left"),
    (202.5, 247.5,  "toward the lower-left"),
    (247.5, 292.5,  "toward the bottom of frame"),
    (292.5, 337.5,  "toward the lower-right"),
]


def speed_to_text(speed_px_per_frame: float) -> str:
    for lo, hi, word in SPEED_VOCAB:
        if lo <= speed_px_per_frame < hi:
            return word
    return "moving"


def direction_to_text(angle_deg: float) -> str:
    angle = angle_deg % 360
    for lo, hi, word in DIRECTION_VOCAB:
        if lo <= 360:
            if lo < hi:
                if lo <= angle < hi:
                    return word
            else:  # wraps around 360
                if angle >= lo or angle < hi:
                    return word
    return "in various directions"


# ═══════════════════════════════════════════════════════════════════
# MOTION FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

class MotionFeatures:
    """Structured motion properties to be routed to LLM Semantic Fusion."""

    def __init__(
        self,
        mean_flow: Tuple[float, float],
        mean_speed: float,
        mean_direction_deg: float,
        acceleration: float,
        flow_maps: Optional[torch.Tensor] = None, # kept optional for memory saving
    ):
        self.mean_flow = mean_flow
        self.mean_speed = mean_speed
        self.mean_direction_deg = mean_direction_deg
        self.acceleration = acceleration
        self.flow_maps = flow_maps

    def to_dict(self) -> Dict[str, Any]:
        """Export to LLM-friendly structured format."""
        
        if self.acceleration < -0.5:
            accel_word = "decelerating"
        elif self.acceleration > 0.5:
            accel_word = "accelerating"
        else:
            accel_word = "at constant speed"

        return {
            "mean_speed_px_frame": round(self.mean_speed, 2),
            "speed_category": speed_to_text(self.mean_speed),
            "direction_deg": round(self.mean_direction_deg, 2),
            "direction_category": direction_to_text(self.mean_direction_deg),
            "acceleration_category": accel_word,
            "acceleration_val": round(self.acceleration, 2),
            "vector_dx": round(self.mean_flow[0], 2),
            "vector_dy": round(self.mean_flow[1], 2),
        }


# ═══════════════════════════════════════════════════════════════════
# MAIN MOTION ANALYZER
# ═══════════════════════════════════════════════════════════════════

class MotionAnalyzer:
    def __init__(
        self,
        method: str = "raft",
        device: str = "cuda",
        downsample_for_flow: float = 0.5,
    ):
        self.method = method
        self.device = device
        self.downsample = downsample_for_flow
        self._raft_model = None

        if method == "raft":
            try:
                from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
                self._raft_model = raft_large(
                    weights=Raft_Large_Weights.DEFAULT
                ).to(device).eval()
                print("[MotionAnalyzer] RAFT optical flow initialized")
            except Exception as e:
                print(f"[MotionAnalyzer] RAFT init failed ({e}); fallback to Farneback")
                self.method = "farneback"

    @torch.no_grad()
    def analyze(
        self,
        history_tensor: torch.Tensor,  # (K, 3, H, W) in [-1, 1]
        bounding_boxes: Optional[List[Tuple[int, int, int, int]]] = None, # list of [x1, y1, x2, y2]
        motion_thresh: float = 1.0,    # Dynamic Thresholding to eliminate static background
    ) -> MotionFeatures:
        K = history_tensor.shape[0]
        if K < 2:
            return self._empty_features()

        orig_h, orig_w = history_tensor.shape[2:]

        # Downsample for speed
        if self.downsample < 1.0:
            h = int(orig_h * self.downsample)
            w = int(orig_w * self.downsample)
            t_resized = F.interpolate(history_tensor, size=(h, w), mode="bilinear", align_corners=False)
        else:
            t_resized = history_tensor
            h, w = orig_h, orig_w

        # Compute flow using Batched RAFT or CPU Fallback
        if self.method == "raft" and self._raft_model is not None:
            flow_stack = self._compute_batch_raft(t_resized) # (K-1, 2, H, W)
        else:
            flow_stack = self._compute_sequence_farneback(t_resized) # (K-1, 2, H, W)

        # ── Dynamic Region & Motion Thresholding ─────────────────────
        
        magnitudes = torch.sqrt(flow_stack[:, 0]**2 + flow_stack[:, 1]**2) # (K-1, H, W)
        
        # Mask setup: if bboxes provided, only look inside bboxes
        mask = torch.zeros_like(magnitudes, dtype=torch.bool)
        if bounding_boxes and len(bounding_boxes) > 0:
            for bbox in bounding_boxes:
                x1, y1, x2, y2 = bbox
                # Adjust bbox coordinates to downsampled resolution
                sx1 = int(x1 * self.downsample)
                sy1 = int(y1 * self.downsample)
                sx2 = int(x2 * self.downsample)
                sy2 = int(y2 * self.downsample)
                mask[:, sy1:sy2, sx1:sx2] = True
        else:
            mask[:] = True

        # Dynamic motion thresholding: only average moving pixels
        moving_mask = mask & (magnitudes > motion_thresh)

        if moving_mask.sum() == 0:
            return self._empty_features()

        # Compute masked means
        mean_dx = flow_stack[:, 0][moving_mask].mean().item()
        mean_dy = flow_stack[:, 1][moving_mask].mean().item()
        mean_speed = magnitudes[moving_mask].mean().item()

        # Scale speed back up to original pixel dimensions
        if self.downsample < 1.0:
            mean_speed /= self.downsample
            mean_dx /= self.downsample
            mean_dy /= self.downsample

        # Direction: Cartesian math (Y is inverted in image space so we use -mean_dy)
        mean_direction_deg = math.degrees(math.atan2(-mean_dy, mean_dx)) % 360

        # Acceleration approximation
        if K > 2:
            first_frame_mag = magnitudes[0][moving_mask[0]].mean().item() if moving_mask[0].sum() > 0 else 0
            last_frame_mag = magnitudes[-1][moving_mask[-1]].mean().item() if moving_mask[-1].sum() > 0 else 0
            acceleration = (last_frame_mag - first_frame_mag) / self.downsample
        else:
            acceleration = 0.0

        return MotionFeatures(
            mean_flow=(mean_dx, mean_dy),
            mean_speed=mean_speed,
            mean_direction_deg=mean_direction_deg,
            acceleration=acceleration,
        )

    def _empty_features(self) -> MotionFeatures:
        return MotionFeatures((0.0, 0.0), 0.0, 0.0, 0.0)

    def _compute_batch_raft(self, t_in: torch.Tensor) -> torch.Tensor:
        """
        Batched RAFT processing with Divisibility-by-8 Padding to prevent crashes.
        t_in: (K, 3, H, W) in [-1, 1]
        """
        # RAFT expects [0, 255]
        t = ((t_in + 1.0) / 2.0 * 255.0).to(self.device)
        
        t1 = t[:-1] # (K-1, 3, H, W)
        t2 = t[1:]  # (K-1, 3, H, W)
        
        _, _, h, w = t1.shape
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        
        # Pad replication for safety
        if pad_h > 0 or pad_w > 0:
            t1 = F.pad(t1, (0, pad_w, 0, pad_h), mode='replicate')
            t2 = F.pad(t2, (0, pad_w, 0, pad_h), mode='replicate')
            
        flow_predictions = self._raft_model(t1, t2)
        flow = flow_predictions[-1] # (K-1, 2, H_pad, W_pad)
        
        # Crop back to exact original size
        if pad_h > 0 or pad_w > 0:
            flow = flow[..., :h, :w]
            
        return flow # (K-1, 2, H, W)

    def _compute_sequence_farneback(self, t_in: torch.Tensor) -> torch.Tensor:
        """Fallback CPU processing."""
        flows = []
        frames = []
        for i in range(t_in.shape[0]):
            arr = ((t_in[i].cpu().float() + 1.0) / 2.0).clamp(0, 1)
            arr = (arr * 255).byte().permute(1, 2, 0).numpy()
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            frames.append(gray)
            
        for i in range(len(frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                frames[i], frames[i + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            # shape (H, W, 2)
            flows.append(torch.from_numpy(flow).permute(2, 0, 1)) # (2, H, W)
            
        return torch.stack(flows, dim=0).to(self.device) # (K-1, 2, H, W)
