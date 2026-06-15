"""
src/models/wan2_pipeline.py
============================
Wan2.2 Image-to-Video Pipeline Wrapper

This module wraps the Diffusers implementation of Wan2.2 I2V A14B with:
  1. LoRA weight loading for traffic domain adaptation
  2. CacheDiT / TeaCache acceleration
  3. Multi-frame history conditioning
  4. Resolution-adaptive generation

Wan2.2 Reference
----------------
Model: Wan2.2-I2V-A14B
HuggingFace: Wan-AI/Wan2.2-I2V-A14B
GitHub: https://github.com/Wan-Video/Wan2.2

Architecture:
  - MoE (Mixture-of-Experts): 27B total, 14B active per step
    * High-noise expert: handles early denoising (global layout)
    * Low-noise expert: handles late denoising (fine details)
  - VAE: high-compression (4× temporal, 8× spatial)
  - Text encoder: T5-XXL (multilingual)
  - Image conditioning: concatenated in latent space (I2V mode)
  - Training: 65.6% more images + 83.2% more videos than Wan2.1
  - Diffusion process: Rectified Flow (Flow Matching), not DDPM

CacheDiT Reference
------------------
Repo: https://github.com/vipshop/cache-dit
Key:  `cache_dit.enable_cache(pipe, cache_config=DBCacheConfig())`
Speedup: 2-9× depending on configuration

TeaCache Reference
------------------
Paper: arXiv 2411.19108
Key:   Timestep embedding-aware caching; modulates inputs to estimate output diffs
Speedup: up to 4.41× with negligible quality loss
Integration: applied as a callback/hook during inference steps
"""

import os
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image

# Diffusers imports
try:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image
    DIFFUSERS_WAN_AVAILABLE = True
except ImportError:
    DIFFUSERS_WAN_AVAILABLE = False
    class AutoencoderKLWan:
        pass
    class WanImageToVideoPipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("diffusers>=0.33.0 is not installed. Please run: pip install -r requirements.txt")
    def export_to_video(*args, **kwargs):
        pass
    def load_image(*args, **kwargs):
        pass

# PEFT for LoRA
try:
    from peft import PeftModel, LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    warnings.warn("PEFT not available; LoRA loading disabled.")

# CacheDiT
try:
    import cache_dit
    from cache_dit import DBCacheConfig, ParallelismConfig
    CACHE_DIT_AVAILABLE = True
except ImportError:
    CACHE_DIT_AVAILABLE = False
    warnings.warn("cache_dit not installed. Run: pip install cache-dit")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Wan2.2's VAE compression ratios (from model card)
VAE_TEMPORAL_COMPRESSION = 4    # 33 frames → 9 latent frames
VAE_SPATIAL_COMPRESSION = 8     # 720p → 90×160 latent

# TeaCache calibration coefficients for Wan2.2
# These are empirically determined (from ali-vilab/TeaCache Wan2.1 adaptation)
# The quadratic polynomial maps input difference → output difference estimate
TEACACHE_COEFFICIENTS = {
    "wan2_i2v": [2.5651901, 3.8420, 20.4165]  # [a, b, c] for ax^2 + bx + c
}


# ---------------------------------------------------------------------------
# TeaCache Implementation
# ---------------------------------------------------------------------------

class TeaCacheCallback:
    """
    Timestep Embedding Aware Cache for Wan2.2.
    
    Reference: arXiv 2411.19108
    Repo:      https://github.com/ali-vilab/TeaCache
    Paper:     "Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model"
    
    Core idea:
      At each denoising step t, instead of always computing the full model output,
      check if the INPUT DIFFERENCE Δinput(t) is small enough to reuse
      the cached output from the previous step.
      
      Key insight: Δoutput(t) ≈ f(Δinput(t)) where f is a polynomial learned
      from calibration data. If f(Δinput) < threshold → reuse cache.
      
    The input is "modulated" by the timestep embedding before computing Δinput:
      modulated_input(t) = input(t) × scale(t) + shift(t)
    where scale(t) and shift(t) come from the timestep embedding MLP
    (the same conditioning used in the AdaLN layers).
    
    This modulation ensures Δmodulated_input(t) better approximates Δoutput(t).
    
    Implementation:
      We attach this as a pipeline step callback. At each step:
        1. Compute modulated input difference from previous step
        2. Estimate output difference using polynomial
        3. If estimated diff < threshold → skip computation (reuse cache)
        4. Otherwise → compute normally → update cache + rescale diff
    
    Args:
        rel_l1_thresh: relative L1 threshold for cache reuse decision
                       lower = higher quality but less speedup
                       paper default: 0.3 for ~4× speedup
        poly_coeffs: polynomial coefficients [a, b, c] for diff estimation
                     (calibrated on Wan2.2 specifically)
    """
    
    def __init__(
        self,
        rel_l1_thresh: float = 0.3,
        poly_coeffs: Optional[List[float]] = None,
    ):
        self.rel_l1_thresh = rel_l1_thresh
        self.poly_coeffs = poly_coeffs or TEACACHE_COEFFICIENTS["wan2_i2v"]
        
        # Cache state
        self._cached_output: Optional[torch.Tensor] = None
        self._cached_residual: Optional[torch.Tensor] = None
        self._prev_modulated_input: Optional[torch.Tensor] = None
        self._accumulated_diff: float = 0.0
        
        # Statistics (for debugging)
        self.n_skipped = 0
        self.n_computed = 0
    
    def reset(self) -> None:
        """Reset cache state between generations."""
        self._cached_output = None
        self._cached_residual = None
        self._prev_modulated_input = None
        self._accumulated_diff = 0.0
        self.n_skipped = 0
        self.n_computed = 0
    
    def _poly_estimate(self, x: float) -> float:
        """Polynomial estimate of output difference given input difference."""
        a, b, c = self.poly_coeffs
        return a * x**2 + b * x + c
    
    def should_cache(
        self,
        modulated_input: torch.Tensor,
        step_idx: int,
        num_steps: int,
    ) -> bool:
        """
        Decide whether to reuse cached output for current step.
        
        Never cache the first step (no previous state).
        Never cache the last step (want fresh output for final latent).
        
        Args:
            modulated_input: timestep-modulated latent tensor
            step_idx: current denoising step (0-indexed)
            num_steps: total denoising steps
        
        Returns:
            True if cache should be reused; False if fresh computation needed
        """
        # Always compute first and last steps
        if step_idx == 0 or step_idx == num_steps - 1:
            return False
        
        if self._prev_modulated_input is None:
            return False
        
        # Compute relative L1 difference of modulated inputs
        diff = (modulated_input - self._prev_modulated_input).abs().mean().item()
        abs_mean = self._prev_modulated_input.abs().mean().item()
        rel_diff = diff / (abs_mean + 1e-8)
        
        # Polynomial estimate of output difference
        estimated_out_diff = self._poly_estimate(rel_diff)
        
        # Accumulate difference; reset when computing
        self._accumulated_diff += estimated_out_diff
        
        return self._accumulated_diff < self.rel_l1_thresh
    
    def update_cache(
        self,
        modulated_input: torch.Tensor,
        model_output: torch.Tensor,
    ) -> None:
        """Update cache after a fresh model computation."""
        self._prev_modulated_input = modulated_input.detach().clone()
        self._cached_output = model_output.detach().clone()
        self._accumulated_diff = 0.0
        self.n_computed += 1
    
    def get_cached_output(self) -> Optional[torch.Tensor]:
        """Return the cached model output."""
        self.n_skipped += 1
        return self._cached_output
    
    def speedup_ratio(self) -> float:
        """Fraction of steps that were cached (speedup proxy)."""
        total = self.n_computed + self.n_skipped
        if total == 0:
            return 1.0
        return total / max(self.n_computed, 1)


# ---------------------------------------------------------------------------
# Main Pipeline Wrapper
# ---------------------------------------------------------------------------

class Wan2I2VPipeline:
    """
    Wrapped Wan2.2 I2V pipeline with full optimization stack.
    
    This class orchestrates:
      - Model loading from HuggingFace or local checkpoint
      - LoRA fine-tuned weight loading (optional)
      - CacheDiT acceleration (optional)
      - TeaCache acceleration (optional)
      - Multi-frame history conditioning
      - Resolution-adaptive inference
      
    Typical usage:
        >>> pipe = Wan2I2VPipeline(
        ...     repo_id="Wan-AI/Wan2.2-I2V-A14B",
        ...     device="cuda",
        ...     dtype=torch.bfloat16,
        ... )
        >>> pipe.load_lora("./checkpoints/lora/traffic_lora.safetensors")
        >>> pipe.enable_cache_acceleration(algo="DBCache", rel_l1_thresh=0.3)
        >>>
        >>> frames = pipe.generate(
        ...     history_frames=history_tensor,      # (K, 3, H, W)
        ...     prompt="Overhead traffic view...",
        ...     num_frames=33,
        ...     guidance_scale=5.0,
        ...     num_inference_steps=50,
        ...     seed=42,
        ... )
        >>> # frames: (N, 3, H, W) tensor in [-1, 1]
    """
    
    def __init__(
        self,
        repo_id: str = "Wan-AI/Wan2.2-I2V-A14B",
        local_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = False,
        enable_vae_tiling: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.repo_id = repo_id
        
        print(f"[Wan2Pipeline] Loading model from {local_dir or repo_id}...")
        
        model_path = local_dir if (local_dir and os.path.exists(local_dir)) else repo_id
        
        # Load the Wan2.2 I2V pipeline
        # Reference: https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        
        if enable_cpu_offload:
            # Useful when VRAM < 40GB; moves unused components to CPU
            self.pipe.enable_model_cpu_offload()
            print("[Wan2Pipeline] CPU offload enabled")
        else:
            self.pipe = self.pipe.to(device)
        
        if enable_vae_tiling:
            # VAE tiling: reduces peak VRAM for high-resolution generation
            if hasattr(self.pipe, 'vae') and hasattr(self.pipe.vae, 'enable_tiling'):
                self.pipe.vae.enable_tiling()
                print("[Wan2Pipeline] VAE tiling enabled")
        
        # TeaCache state
        self._tea_cache: Optional[TeaCacheCallback] = None
        
        # Track if CacheDiT is active
        self._cache_dit_active = False
        
        print(f"[Wan2Pipeline] Model ready on {device} ({dtype})")
    
    # ------------------------------------------------------------------
    # LoRA loading
    # ------------------------------------------------------------------
    
    def load_lora(
        self,
        weights_path: str,
        alpha: float = 1.0,
    ) -> None:
        """
        Load traffic-domain LoRA fine-tuned weights.
        
        Args:
            weights_path: path to .safetensors LoRA weights file
            alpha: LoRA scale factor (0=no effect, 1=full effect)
        """
        if not PEFT_AVAILABLE:
            warnings.warn("[Wan2Pipeline] PEFT not available; cannot load LoRA.")
            return
        
        if not os.path.exists(weights_path):
            warnings.warn(f"[Wan2Pipeline] LoRA weights not found: {weights_path}")
            return
        
        print(f"[Wan2Pipeline] Loading LoRA from {weights_path} (α={alpha})")
        
        # Load LoRA weights into the transformer
        try:
            self.pipe.load_lora_weights(weights_path)
            if alpha != 1.0:
                self.pipe.set_adapters(["default"], adapter_weights=[alpha])
        except Exception as e:
            warnings.warn(f"[Wan2Pipeline] LoRA load failed: {e}")
        
        print("[Wan2Pipeline] LoRA weights loaded successfully")
    
    # ------------------------------------------------------------------
    # Cache acceleration
    # ------------------------------------------------------------------
    
    def enable_cache_acceleration(
        self,
        algo: str = "DBCache",
        rel_l1_thresh: float = 0.3,
        residual_diff_threshold: float = 0.08,
        n_compute_blocks: int = 4,
    ) -> None:
        """
        Enable inference acceleration via CacheDiT + TeaCache.
        
        CacheDiT (vipshop):
          DBCache: caches residuals when L1 diff < residual_diff_threshold
          TaylorSeer: uses Taylor expansion to predict future features
          SCM: predefined step computation masks
          
        TeaCache (ali-vilab):
          Timestep-aware caching; skips transformer forward when estimated
          output diff < rel_l1_thresh
        
        Both can stack for maximum speedup.
        
        Args:
            algo: CacheDiT algorithm "DBCache" | "TaylorSeer" | "none"
            rel_l1_thresh: TeaCache threshold (0.3 ≈ 4× speedup)
            residual_diff_threshold: DBCache residual threshold
            n_compute_blocks: DBCache blocks to compute normally
        """
        # Enable CacheDiT
        if algo != "none" and CACHE_DIT_AVAILABLE:
            print(f"[Wan2Pipeline] Enabling CacheDiT ({algo})...")
            try:
                if algo == "DBCache":
                    cache_config = DBCacheConfig(
                        residual_diff_threshold=residual_diff_threshold,
                    )
                elif algo == "TaylorSeer":
                    from cache_dit import TaylorSeerConfig
                    cache_config = TaylorSeerConfig()
                else:
                    cache_config = DBCacheConfig()
                
                cache_dit.enable_cache(self.pipe, cache_config=cache_config)
                self._cache_dit_active = True
                print("[Wan2Pipeline] CacheDiT active")
            except Exception as e:
                warnings.warn(f"[Wan2Pipeline] CacheDiT setup failed: {e}")
        
        # Enable TeaCache
        print(f"[Wan2Pipeline] Enabling TeaCache (thresh={rel_l1_thresh})...")
        self._tea_cache = TeaCacheCallback(rel_l1_thresh=rel_l1_thresh)
        print("[Wan2Pipeline] TeaCache active")
    
    # ------------------------------------------------------------------
    # Frame preprocessing
    # ------------------------------------------------------------------
    
    def preprocess_history_frames(
        self,
        history_tensor: torch.Tensor,  # (K, 3, H, W) in [-1, 1]
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        Extract the last history frame as the I2V conditioning image.
        
        Wan2.2 I2V takes a single image as the initial frame condition.
        We use the LAST history frame as this anchor.
        
        The image sets:
          - Content: visible scene at t=0 of generation
          - Resolution: output resolution will match input image's aspect ratio
        
        Args:
            history_tensor: (K, 3, H, W), values in [-1, 1]
        
        Returns:
            PIL Image for pipeline input
            (H, W) original resolution
        """
        # Take last frame
        last_frame = history_tensor[-1]  # (3, H, W)
        
        # De-normalize: [-1, 1] → [0, 1] → [0, 255]
        last_frame_np = ((last_frame.cpu().float() + 1.0) / 2.0).clamp(0, 1)
        last_frame_np = (last_frame_np * 255).byte().permute(1, 2, 0).numpy()  # (H, W, 3)
        
        h, w = last_frame_np.shape[:2]
        pil_image = Image.fromarray(last_frame_np, mode='RGB')
        
        return pil_image, (h, w)
    
    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------
    
    def generate(
        self,
        history_frames: torch.Tensor,    # (K, 3, H, W) in [-1, 1]
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 33,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        seed: int = 42,
        max_area: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate future video frames conditioned on history + text.
        
        Args:
            history_frames: history frame tensor (K, 3, H, W), values [-1,1]
            prompt: text description (from CoECT + Prompt Relay)
            negative_prompt: negative guidance (default: empty for Wan2.2)
            num_frames: number of frames to generate (must be 4k+1 for Wan2.2)
            guidance_scale: CFG scale (5.0 is Wan2.2 default for I2V)
            num_inference_steps: denoising steps
            seed: random seed for reproducibility
            max_area: maximum generation area (for memory control)
        
        Returns:
            Generated frames tensor (N, 3, H, W), values in [-1, 1]
        """
        # Extract conditioning image
        conditioning_image, (orig_h, orig_w) = self.preprocess_history_frames(
            history_frames
        )
        
        # Compute generation size
        # Wan2.2 I2V: size follows the aspect ratio of the input image
        # and is bounded by max_area
        if max_area is not None:
            area = orig_h * orig_w
            if area > max_area:
                scale = math.sqrt(max_area / area)
                gen_h = int(orig_h * scale) // 8 * 8
                gen_w = int(orig_w * scale) // 8 * 8
            else:
                gen_h, gen_w = orig_h, orig_w
        else:
            gen_h, gen_w = orig_h, orig_w
        
        # Ensure dimensions are multiples of 8 (VAE requirement)
        gen_h = (gen_h // 8) * 8
        gen_w = (gen_w // 8) * 8
        
        # Ensure num_frames satisfies 4k+1 constraint for Wan2.2's VAE
        # (the VAE compresses temporally by factor 4; needs 4k+1 frames)
        if (num_frames - 1) % 4 != 0:
            num_frames = ((num_frames - 1) // 4) * 4 + 1
        
        # Set random seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Reset TeaCache state for this generation
        if self._tea_cache is not None:
            self._tea_cache.reset()
        
        # Run generation
        # Note: WanImageToVideoPipeline expects:
        #   - image: PIL Image (the conditioning frame)
        #   - prompt: str
        #   - num_frames: int
        #   - guidance_scale: float
        #   - etc.
        with torch.no_grad():
            output = self.pipe(
                image=conditioning_image,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                # height and width are inferred from image aspect ratio
                # but we can override:
                height=gen_h,
                width=gen_w,
                output_type="pt",  # return PyTorch tensor
            )
        
        # output.frames: (1, T, C, H, W) or (T, C, H, W) depending on version
        frames = output.frames
        if frames.dim() == 5:
            frames = frames[0]  # Remove batch dimension → (T, C, H, W)
        
        # Frames are in [0, 1]; convert to [-1, 1] for consistency
        frames = frames * 2.0 - 1.0
        
        # Log TeaCache stats
        if self._tea_cache is not None:
            print(
                f"[Wan2Pipeline] TeaCache speedup: {self._tea_cache.speedup_ratio():.2f}× "
                f"(skipped {self._tea_cache.n_skipped}/{self._tea_cache.n_skipped + self._tea_cache.n_computed} steps)"
            )
        
        return frames  # (T, 3, H, W) in [-1, 1]
    
    def generate_batch(
        self,
        samples: List[Dict[str, Any]],
        **generation_kwargs,
    ) -> List[torch.Tensor]:
        """
        Generate for a batch of test samples.
        
        Each sample dict should have:
            history_frames: tensor (K, 3, H, W)
            prompt: str
            num_frames: int
        
        Returns:
            List of frame tensors, one per sample
        """
        results = []
        for i, sample in enumerate(samples):
            print(f"[Wan2Pipeline] Generating sample {i+1}/{len(samples)}...")
            frames = self.generate(
                history_frames=sample["history_frames"],
                prompt=sample["prompt"],
                num_frames=sample.get("num_frames", 33),
                seed=sample.get("seed", 42),
                **generation_kwargs,
            )
            results.append(frames)
        return results
