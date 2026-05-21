"""
Mask-Based Motion Trajectory Prediction Sub-Network

Predicts 2D spatio-temporal heatmaps defining the future pixel locations
of pedestrians and vehicles over the forecasted horizon. This trajectory
acts as an explicit intermediate representation that guides the DiT to
synthesize subjects along predefined mathematical paths.

Ensures physical movement aligns with semantic text instructions,
eliminating random structural morphing and improving FVD/CLIP-S scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TrajectoryEncoder(nn.Module):
    """
    Encodes history latents and text embeddings into trajectory features.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        text_dim: int = 768,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_proj = nn.Sequential(
            nn.Conv3d(latent_dim, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        history_latents: torch.Tensor,
        text_pedestrian: torch.Tensor,
        text_vehicle: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode history and text into trajectory features.

        Args:
            history_latents: [B, C_latent, T_hist, H, W]
            text_pedestrian: [B, L_p, D]
            text_vehicle: [B, L_v, D]

        Returns:
            features: [B, hidden_dim, T_hist, H, W]
        """
        latent_feat = self.latent_proj(history_latents)

        text_combined = (text_pedestrian + text_vehicle) / 2
        text_feat = self.text_proj(text_combined.mean(dim=1))

        B, D, T, H, W = latent_feat.shape
        text_expanded = text_feat.view(B, -1, 1, 1, 1).expand(-1, -1, T, H, W)

        fused = torch.cat([latent_feat, text_expanded], dim=1)
        return self.fusion(fused)


class TrajectoryDecoder(nn.Module):
    """
    Decodes trajectory features into spatio-temporal heatmaps.

    Outputs separate heatmaps for pedestrian and vehicle trajectories
    over the full future horizon.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_future_frames: int = 8,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_future_frames = num_future_frames
        self.num_classes = num_classes

        self.temporal_expand = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(
                hidden_dim, hidden_dim // 2,
                kernel_size=(2, 4, 4), stride=(2, 4, 4),
            ),
            nn.GroupNorm(32, hidden_dim // 2),
            nn.SiLU(),
            nn.ConvTranspose3d(
                hidden_dim // 2, hidden_dim // 4,
                kernel_size=(2, 4, 4), stride=(2, 4, 4),
            ),
            nn.GroupNorm(32, hidden_dim // 4),
            nn.SiLU(),
        )

        self.heatmap_head = nn.Conv3d(
            hidden_dim // 4, num_classes * num_future_frames,
            kernel_size=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features into trajectory heatmaps.

        Args:
            features: [B, hidden_dim, T_hist, H_latent, W_latent]

        Returns:
            heatmaps: [B, num_classes, num_future, H, W]
        """
        h = self.temporal_expand(features)
        h = self.upsample(h)
        h = self.heatmap_head(h)

        B, _, T, H, W = h.shape
        h = h.view(B, self.num_classes, self.num_future_frames, T, H, W)

        h = h.mean(dim=3)

        return h


class MotionTrajectoryPredictor(nn.Module):
    """
    Complete motion trajectory prediction network.

    Takes history latents and dual text embeddings, outputs
    spatio-temporal heatmaps for pedestrian and vehicle trajectories.

    These heatmaps are concatenated as additional conditional input
    into the DiT via masked spatio-temporal self-attention.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        text_dim: int = 768,
        hidden_dim: int = 256,
        num_future_frames: int = 8,
        num_classes: int = 2,
    ):
        super().__init__()
        self.encoder = TrajectoryEncoder(latent_dim, text_dim, hidden_dim)
        self.decoder = TrajectoryDecoder(hidden_dim, num_future_frames, num_classes)
        self.num_classes = num_classes
        self.num_future_frames = num_future_frames

    def forward(
        self,
        history_latents: torch.Tensor,
        text_pedestrian: torch.Tensor,
        text_vehicle: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict motion trajectory heatmaps.

        Args:
            history_latents: [B, C_latent, T_hist, H, W]
            text_pedestrian: [B, L_p, D]
            text_vehicle: [B, L_v, D]

        Returns:
            heatmaps: [B, num_classes, num_future, H_out, W_out]
        """
        features = self.encoder(history_latents, text_pedestrian, text_vehicle)
        heatmaps = self.decoder(features)
        return heatmaps

    def compute_loss(
        self,
        history_latents: torch.Tensor,
        text_pedestrian: torch.Tensor,
        text_vehicle: torch.Tensor,
        target_heatmaps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute trajectory prediction loss.

        Args:
            target_heatmaps: [B, num_classes, num_future, H, W] - ground truth

        Returns:
            loss: BCE + focal loss combination
        """
        pred = self.forward(history_latents, text_pedestrian, text_vehicle)

        bce_loss = F.binary_cross_entropy_with_logits(pred, target_heatmaps)

        pred_sigmoid = torch.sigmoid(pred)
        focal_weight = (1 - pred_sigmoid) ** 2
        focal_loss = F.binary_cross_entropy_with_logits(
            pred, target_heatmaps, reduction="none"
        )
        focal_loss = (focal_weight * focal_loss).mean()

        return bce_loss + focal_loss


class TrajectoryConditioningModule(nn.Module):
    """
    Integrates trajectory heatmaps into DiT processing.

    Concatenates trajectory heatmaps as additional channels
    to the DiT input and applies masked spatio-temporal attention.
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_future_frames: int = 8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_future_frames = num_future_frames

        self.trajectory_proj = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        trajectory_heatmaps: torch.Tensor,
        t_index: int = 0,
    ) -> torch.Tensor:
        """
        Apply trajectory conditioning to DiT input.

        Args:
            x: [B, C, T, H, W] - DiT input latents
            trajectory_heatmaps: [B, num_classes, num_future, H, W]
            t_index: current frame index within the future horizon

        Returns:
            x_conditioned: [B, C+1, T, H, W] - conditioned input
        """
        if trajectory_heatmaps is None:
            B, C, T, H, W = x.shape
            return torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)

        traj_t = trajectory_heatmaps[:, :, t_index]
        traj_mask = self.trajectory_proj(traj_t.mean(dim=1, keepdim=True))

        if traj_mask.dim() == 4:
            traj_mask = traj_mask.unsqueeze(2)

        if traj_mask.shape[-2:] != x.shape[-2:]:
            traj_mask = F.interpolate(
                traj_mask.squeeze(2),
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(2)

        x_conditioned = torch.cat([x, traj_mask], dim=1)
        return x_conditioned


class MaskedSpatioTemporalAttention(nn.Module):
    """
    Spatio-temporal self-attention modulated by trajectory masks.

    Forces the DiT to synthesize subjects along the predefined
    trajectory path, preventing random structural morphing.
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        num_heads: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.mask_proj = nn.Sequential(
            nn.Linear(1, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_heads),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        trajectory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply masked spatio-temporal attention.

        Args:
            x: [B, N, D] - flattened spatio-temporal tokens
            trajectory_mask: [B, N] - trajectory attention weights

        Returns:
            out: [B, N, D]
        """
        B, N, D = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if trajectory_mask is not None:
            mask_bias = self.mask_proj(trajectory_mask.unsqueeze(-1))
            mask_bias = mask_bias.mean(dim=-1).view(B, 1, 1, N)
            attn = attn + mask_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)

        return out
