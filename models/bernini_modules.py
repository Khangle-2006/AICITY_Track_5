"""
src/models/bernini_modules.py
==============================
Bernini-Inspired Modules for WTS Track 5
arXiv: 2605.22344 — "Bernini: Latent Semantic Planning for Video Diffusion"
ByteDance, May 2026

EXTRACTED IDEAS & ADAPTATIONS:
══════════════════════════════════════════════════════════════════════════════

[IDEA 1] SA-3D RoPE (Segment-Aware 3D Rotary Position Embedding)
  Paper:   §2.1.2, Eq.(2): r̃_{t,h,w,i} = r_{t,h,w} ⊙ r^seg_i
  Problem: When history frames + ped tokens + veh tokens are concatenated,
           they share identical (t,h,w) coordinates → attention identity
           ambiguity → model cannot distinguish which input segment each
           token belongs to.
  Fix:     Assign each visual/text segment an index i. Construct a
           full-dimensional rotary frequency vector r^seg_i that encodes i.
           Multiply element-wise with standard RoPE frequencies.
           Different i → different global phase offset → distinguishable.

  Our WTS Segment Map:
    i=0  target (future frames, masked during training, generated at inference)
    i=1  history frames (visual conditioning from K past frames)
    i=2  pedestrian caption tokens
    i=3  vehicle caption tokens

  Implementation: We apply a learnable per-segment phase offset to the
  RoPE frequencies of Wan2.2's DiT attention layers, injected as a
  forward pre-hook. No retraining of the base model required for inference;
  the offset can be trained during LoRA fine-tuning.

[IDEA 2] Source VAE Feature Injection ("Latent Inpainting Mask")
  Paper:   §2.1.2 — "VAE features of the source input are additionally
           injected to preserve detail and consistency."
  Problem: Conditioning on a single PIL image (last history frame) loses
           multi-frame temporal context and fine-grained texture detail.
  Fix:     Encode all K history frames with the VAE → get z_hist.
           Concatenate z_hist with the noisy latent z_t along channel dim.
           Add a binary inpainting mask (1=history/known, 0=future/unknown).
           Linear projection 2C+1 → C to restore original model channel dim.
           This is the "ControlNet-lite" / "inpainting" paradigm.

  Architecture:
    z_hist = VAE.encode(history_frames)          # (B, C, T_h, H', W')
    mask   = build_inpaint_mask(T_h, T_target)  # (B, 1, T, H', W')
    z_pad  = pad_or_trim(z_hist, T_target)      # (B, C, T, H', W')
    z_in   = concat([z_noisy, z_pad, mask], 1)  # (B, 2C+1, T, H', W')
    z_in   = self.proj(z_in)                    # (B, C, T, H', W')

[IDEA 3] Chain-of-Thought Semantic Planning
  Paper:   §2.1.1 — "chain-of-thought reasoning in the planner to better
           transfer understanding into generation"
  Implementation: Already in CoECTProcessor via allow_chain_of_thought=True.
  We strip <think>...</think> tags before JSON parsing (Bug Fix #6).

[IDEA 4] Segment-wise Hybrid Attention Mask
  Paper:   Figure 3(D) — source segments can attend to each other;
           target attends to all; source cannot attend to target.
  Adaptation: In PromptRelayRouter, ped and veh token ranges get additional
  cross-segment attention penalty (they should not dilute each other).
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# [IDEA 1] SA-3D RoPE
# =============================================================================
class SA3DRoPE(nn.Module):
    """
    Segment-Aware 3D Rotary Position Embedding.

    Bernini §2.1.2, Eq.(2): r̃_{t,h,w,i} = r_{t,h,w} ⊙ r^seg_i

    We implement r^seg_i as a learnable per-segment phase offset:
        r^seg_i = exp(j * θ_i),  θ_i ∈ R^{dim}  (learned during LoRA training)
    
    Applied as a multiplicative phase modulation on top of standard RoPE.
    At inference without LoRA: uses fixed default phases (no learning needed).

    Args:
        dim:        RoPE frequency dimension (must match DiT head_dim)
        n_segments: number of distinct input segments (default 4 for WTS)
        phase_scale: scale of initial random phases (0 → identity; 0.3 → ~17°)
        learnable:  whether to treat phases as learnable parameters
    """

    def __init__(
        self,
        dim: int = 128,
        n_segments: int = 4,
        phase_scale: float = 0.3,
        learnable: bool = True,
    ):
        super().__init__()
        self.n_segments = n_segments
        self.dim = dim

        # Initialize segment phase offsets
        # Segment 0 (target): zero offset (anchor)
        # Segments 1,2,3: linearly spaced phases  [π/4, π/2, 3π/4]
        init_phases = torch.zeros(n_segments, dim)
        for i in range(1, n_segments):
            init_phases[i] = phase_scale * (math.pi * i / n_segments)

        if learnable:
            self.phases = nn.Parameter(init_phases)
        else:
            self.register_buffer("phases", init_phases)

    def get_phase_modulation(self, seg_idx: int, device: torch.device) -> torch.Tensor:
        """
        Get complex phase modulation for segment seg_idx.

        Returns (dim,) complex tensor: exp(j * θ_i)
        In real representation: (cos(θ), sin(θ)) interleaved.
        """
        theta = self.phases[seg_idx].to(device)    # (dim,)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        # Rotate: (x, y) → (x*cos - y*sin, x*sin + y*cos)
        # For RoPE, we return (cos, sin) pairs for downstream rotation
        return cos_t, sin_t

    def apply_to_rope(
        self,
        freqs: torch.Tensor,   # (..., dim) RoPE frequencies (cos or sin)
        seg_idx: int,
    ) -> torch.Tensor:
        """
        Apply segment-aware phase modulation to RoPE frequencies.

        Bernini Eq.(2): r̃ = r ⊙ r^seg_i
        In practice: cos and sin components are phase-rotated.
        """
        if seg_idx == 0:
            return freqs   # Target segment: no modification (anchor)

        cos_mod, sin_mod = self.get_phase_modulation(seg_idx, freqs.device)
        # Rotate the frequency vector by segment phase
        # Treat freqs as complex: Re + j*Im (split at dim//2)
        d = freqs.shape[-1]
        re = freqs[..., :d // 2]
        im = freqs[..., d // 2:]
        cos_m = cos_mod[:d // 2].view(*([1] * (freqs.dim() - 1)), d // 2)
        sin_m = sin_mod[:d // 2].view(*([1] * (freqs.dim() - 1)), d // 2)
        new_re = re * cos_m - im * sin_m
        new_im = re * sin_m + im * cos_m
        return torch.cat([new_re, new_im], dim=-1)


# =============================================================================
# [IDEA 2] Source VAE Feature Injection (Latent Inpainting)
# =============================================================================
class SourceVAEInjection(nn.Module):
    """
    Source VAE Feature Injection for Latent Inpainting Conditioning.

    Bernini §2.1.2: "VAE features of the source input are additionally
    injected to preserve detail and consistency."

    Implementation for WTS:
        1. Encode K history frames with Wan2.2's VAE → z_hist (B, C, T_h, H', W')
        2. Pad/interpolate z_hist to match target temporal length T
        3. Build binary inpainting mask M (1=history, 0=future to generate)
        4. Concatenate: z_input = [z_noisy | z_hist_padded | M] along channels
        5. Linear projection: (2C + 1) → C to restore model channel dim

    During inference: this module processes the conditioning before the
    denoising loop. The projected z_input replaces z_noisy as the loop input.

    During training: z_hist comes from ground-truth history; model learns
    to use this as a rich conditioning signal.

    Args:
        in_channels:  Wan2.2 latent channels (typically 16 for TI2V-5B VAE)
        out_channels: same as in_channels (projection back to model dim)
    """

    def __init__(self, in_channels: int = 16, out_channels: int = 16):
        super().__init__()
        self.in_ch  = in_channels
        self.out_ch = out_channels
        # Project [z_noisy | z_hist | mask] → model dim
        # Input channels: C + C + 1 = 2C + 1
        self.proj = nn.Conv3d(
            in_channels=2 * in_channels + 1,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            for i in range(out_channels):
                self.proj.weight[i, i, 0, 0, 0] = 1.0
        # Starts as identity pass-through (z_noisy unchanged)
        # Gradually learns to use history latents during fine-tuning

    @torch.no_grad()
    def encode_history(
        self,
        history_frames: torch.Tensor,   # (K, 3, H, W) in [-1, 1] on CPU
        vae,                             # Wan2.2 AutoencoderKLWan instance
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode K history frames into latent space.

        Returns:
            z_hist: (1, C, T_h, H', W') float32
            where H' = H/16, W' = W/16, T_h = (K-1)//4 + 1 (VAE stride)
        """
        # (K, 3, H, W) → (1, 3, K, H, W)
        frames = history_frames.unsqueeze(0).to(device=device, dtype=torch.float32)
        # VAE runs in float32 (Wan2.2 requirement)
        with torch.no_grad():
            # Wan2.2 VAE expects (B, C, T, H, W)
            posterior = vae.encode(frames)
            z = posterior.latent_dist.sample()   # (1, C, T_h, H', W')
        return z

    def forward(
        self,
        z_noisy: torch.Tensor,           # (B, C, T, H', W') current noisy latent
        z_hist:  torch.Tensor,           # (B_hist, C, T_h, H', W') history latents
        n_hist_frames: int,              # K: how many latent frames are history
    ) -> torch.Tensor:
        """
        Inject history latents into the noisy latent via concatenation + projection.

        Args:
            z_noisy:       noisy latent at current denoising step
            z_hist:        encoded history latent (may have different T or batch)
            n_hist_frames: number of latent frames that are "known" (history)

        Returns:
            Projected latent (B, C, T, H', W') — same shape as z_noisy
        """
        B, C, T, H, W = z_noisy.shape
        device = z_noisy.device
        dtype  = z_noisy.dtype

        # ── Align batch size of z_hist to B (for classifier-free guidance) ──
        if z_hist.shape[0] != B:
            if B % z_hist.shape[0] == 0:
                z_hist = z_hist.repeat(B // z_hist.shape[0], 1, 1, 1, 1)
            else:
                z_hist = z_hist.expand(B, -1, -1, -1, -1)

        # ── Pad or trim z_hist to match T ──────────────────────────────────
        T_h = z_hist.shape[2]
        if T_h < T:
            # Pad future region with zeros (unknown)
            pad = torch.zeros(B, C, T - T_h, H, W, device=device, dtype=dtype)
            z_hist_padded = torch.cat([z_hist.to(dtype=dtype), pad], dim=2)
        elif T_h > T:
            z_hist_padded = z_hist[:, :, :T].to(dtype=dtype)
        else:
            z_hist_padded = z_hist.to(dtype=dtype)

        # ── Build inpainting mask ───────────────────────────────────────────
        # 1 = history (known), 0 = future (to generate)
        mask = torch.zeros(B, 1, T, H, W, device=device, dtype=dtype)
        mask[:, :, :n_hist_frames] = 1.0

        # ── Concatenate and project ─────────────────────────────────────────
        # [z_noisy | z_hist_padded | mask] → (B, 2C+1, T, H, W)
        z_concat = torch.cat([z_noisy, z_hist_padded, mask], dim=1)
        return self.proj(z_concat)   # (B, C, T, H, W)


# =============================================================================
# BERNINI INPAINTING WRAPPER — inference-time helper
# =============================================================================
class BerniniConditioner:
    """
    High-level wrapper combining SA-3D RoPE + Source VAE Injection.

    Used in run_inference.py to prepare conditioning before the diffusion loop.

    Segment assignment for WTS:
        seg_id=1: history frames latents   (visual context)
        seg_id=2: future target frames     (to generate, starts masked)
    """

    def __init__(
        self,
        vae_injection: Optional[SourceVAEInjection] = None,
        sa3d_rope:     Optional[SA3DRoPE]           = None,
    ):
        self.vae_inj = vae_injection
        self.sa3d    = sa3d_rope

    def prepare(
        self,
        z_noisy:       torch.Tensor,     # (B, C, T, H', W')
        history_frames: torch.Tensor,    # (K, 3, H, W) in [-1,1]
        vae,                             # Wan2.2 VAE
        device:        torch.device,
    ) -> torch.Tensor:
        """
        Apply VAE injection conditioning.
        SA-3D RoPE is applied separately as a hook in the DiT attention layers.
        """
        if self.vae_inj is None:
            return z_noisy

        K = history_frames.shape[0]
        # Encode history
        z_hist = self.vae_inj.encode_history(history_frames, vae, device)

        # Compute T_h (number of latent frames from K history frames)
        # Wan2.2 temporal compression: T_latent = (T_video - 1) // 4 + 1
        T_h = (K - 1) // 4 + 1

        # Inject
        return self.vae_inj(z_noisy, z_hist, T_h)
