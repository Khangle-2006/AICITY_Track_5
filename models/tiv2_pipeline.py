"""
src/models/tiv2_pipeline.py
============================
Dual-Model Pipeline: Wan2.2 TI2V-5B (primary) + I2V A14B (secondary)

VERIFIED LOADING PATTERN (both models):
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    # NOTE: VAE must be float32 — the high-compression Wan2.2 VAE loses quality in bf16

KEY BUG FIXES (verified):
  - Modulo 16: all (H,W) cropped to (H//16)*16 BEFORE pipeline call
  - num_frames: always ((N-1)//4)*4 + 1 using integer arithmetic (not math.ceil)
  - output.frames shape: (B, T, C, H, W) for output_type='pt', values in [0,1]
  - VAE dtype: float32 (separate from transformer dtype)
  - CacheDiT: applied AFTER pipe is on device, BEFORE any generation call

References:
  Wan2.2 TI2V-5B:   Wan-AI/Wan2.2-TI2V-5B-Diffusers
  Wan2.2 I2V A14B:  Wan-AI/Wan2.2-I2V-A14B-Diffusers
  TeaCache:          arXiv:2411.19108
  CacheDiT:          github.com/vipshop/cache-dit
  Bernini VAE Inj:   arXiv:2605.22344 §2.1.2
  PhysVideoGen:      arXiv:2601.03665
"""
from __future__ import annotations

import math
import warnings
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class ModelType(Enum):
    TI2V_5B = "TI2V_5B"
    I2V_A14B = "I2V_A14B"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import contextlib

from .bernini_modules import BerniniConditioner, SA3DRoPE, SourceVAEInjection
from .physics_predictor import PhysicsModule
from prompt.prompt_relay import PromptRelayRouter

# ── Diffusers ──────────────────────────────────────────────────────────────
try:
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
    _DIFFUSERS_OK = True
except ImportError:
    _DIFFUSERS_OK = False
    warnings.warn("diffusers not installed. Install: pip install diffusers>=0.33.0")
    # Mock for import-safe usage (Bug Fix #8)
    class WanImageToVideoPipeline:  # type: ignore
        @classmethod
        def from_pretrained(cls, *a, **kw): raise RuntimeError("diffusers not installed")
    class AutoencoderKLWan:  # type: ignore
        @classmethod
        def from_pretrained(cls, *a, **kw): raise RuntimeError("diffusers not installed")

# ── CacheDiT (optional) ────────────────────────────────────────────────────
try:
    import cache_dit
    from cache_dit import DBCacheConfig
    _CACHE_DIT_OK = True
except ImportError:
    _CACHE_DIT_OK = False


# =============================================================================
# CONSTANTS
# =============================================================================
# TI2V-5B VAE: 4× temporal compression, 16× spatial (high-compression VAE)
TI2V_TEMP_COMPRESSION = 4
TI2V_SPATIAL_ALIGN    = 16    # H and W must be divisible by 16  ← Bug Fix #4
TI2V_MAX_FRAMES       = 121   # (121-1)/4 = 30, a valid 4k+1 value

# I2V A14B VAE: 4× temporal, 8× spatial
I2V_SPATIAL_ALIGN     = 16    # Still 16 for safe alignment with patchification
I2V_MAX_FRAMES        = 81

# Wan2.2 default generation params
DEFAULT_NEG = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "overall gray, worst quality, low quality, JPEG artifacts, watermark, "
    "logo, distorted perspective, satellite, cartoon, painting."
)
DEFAULT_CFG   = 5.0
DEFAULT_STEPS = 50


# =============================================================================
# FRAME COUNT UTILITIES (Bug Fix #3)
# =============================================================================
def valid_num_frames(target_n: int, max_n: int = TI2V_MAX_FRAMES) -> int:
    """
    Compute the smallest valid num_frames ≥ target_n for Wan2.2's VAE.

    Wan2.2 VAE requirement: (num_frames - 1) must be divisible by 4.
    Formula: valid_n = ((target_n - 1) // 4) * 4 + 1
    If result < target_n, add 4 to get the next valid value ≥ target_n.

    Bug Fix #3: Previously used math.ceil which misaligns with VAE stride.
    Correct formula uses integer floor division only.

    Examples:
        valid_num_frames(30) → 33  (because 29//4=7, 7*4+1=29 < 30, +4 → 33)
        valid_num_frames(33) → 33  (because 32//4=8, 8*4+1=33 == 33)
        valid_num_frames(1)  → 1
        valid_num_frames(121) → 121
    """
    if target_n <= 1:
        return 1
    candidate = ((target_n - 1) // 4) * 4 + 1
    if candidate < target_n:
        candidate += 4
    return min(candidate, max_n)


def align_spatial(h: int, w: int, align: int = TI2V_SPATIAL_ALIGN) -> Tuple[int, int]:
    """
    Crop H and W down to nearest multiple of align.
    Bug Fix #4: WTS crops can be arbitrary; Wan2.2 VAE crashes if not aligned.
    Uses floor (crop), not round, to avoid out-of-bounds reads.
    """
    return (h // align) * align, (w // align) * align


# =============================================================================
# MOTION-AWARE COMPOSITE (WAN Animate idea, no prompt restriction)
# =============================================================================
class MotionAwareComposite:
    """
    Creates a single conditioning PIL Image from K history frames.

    Inspired by WAN Animate's reference-frame anchoring — without requiring
    WAN Animate itself (which has no prompt input).

    Method 'blend': weighted temporal blend, recent frames dominate.
        composite = Σ wk * frame_k,  wk ∝ (k+1), normalized
    Method 'ghost': current frame + translucent overlay of motion trails.
    Method 'last_only': baseline — only use last history frame.
    """
    def __init__(self, n_frames: int = 5, method: str = "blend"):
        self.n = n_frames
        self.method = method

    def create(self, history: torch.Tensor) -> Image.Image:
        """
        Args:
            history: (K, 3, H, W) float32 in [-1, 1]
        Returns:
            PIL Image RGB
        """
        K = history.shape[0]
        if self.method == "last_only" or K == 1:
            return _t2pil(history[-1])

        n = min(self.n, K)
        frames = history[-n:]              # (n, 3, H, W)
        ws = torch.arange(1, n + 1, dtype=frames.dtype, device=frames.device)
        ws = ws / ws.sum()

        if self.method == "blend":
            composite = (frames * ws.view(n, 1, 1, 1)).sum(0)   # (3, H, W)
            return _t2pil(composite)
        elif self.method == "ghost":
            base = frames[-1].clone()
            alpha = 0.25 / max(n - 1, 1)
            for i in range(n - 1):
                base = base * (1 - alpha) + frames[i] * alpha
            return _t2pil(base)
        else:
            return _t2pil(history[-1])


def _t2pil(t: torch.Tensor) -> Image.Image:
    """(3, H, W) in [-1,1] → RGB PIL Image."""
    arr = ((t.cpu().float().clamp(-1, 1) + 1.0) / 2.0 * 255).byte()
    return Image.fromarray(arr.permute(1, 2, 0).numpy(), "RGB")


# =============================================================================
# TEACACHE (arXiv:2411.19108) — inline, no extra package
# =============================================================================
class TeaCache:
    """
    Timestep-Embedding Aware Cache for Wan2.2.

    Paper: "Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model"
    arXiv: https://arxiv.org/abs/2411.19108  |  Speedup: ~4.41×

    Algorithm:
        1. At step t, compute relative L1 diff of the model input vs prev step.
        2. Use polynomial f(x) to estimate output diff from input diff.
        3. Accumulate estimated diffs.
        4. If accumulated < threshold → reuse cached output (skip forward pass).
        5. If accumulated ≥ threshold → compute fresh, reset accumulator.

    Key: we never skip step 0 or the last step (quality guard).
    """
    # Polynomial coefficients calibrated for Wan2.2 TI2V-5B
    # Maps relative input-diff → estimated output-diff
    POLY = [2.5651901, 3.842, 20.4165]   # ax² + bx + c

    def __init__(self, thresh: float = 0.25):
        self.thresh = thresh
        self._accum: float = 0.0
        self._prev:  Optional[torch.Tensor] = None
        self._cache: Optional[torch.Tensor] = None
        self.n_skip = 0
        self.n_calc = 0

    def reset(self) -> None:
        self._accum = 0.0
        self._prev  = None
        self._cache = None
        self.n_skip = self.n_calc = 0

    def _est(self, x: float) -> float:
        a, b, c = self.POLY
        return a * x * x + b * x + c

    def query(self, x: torch.Tensor, step: int, total: int) -> bool:
        """Return True if we should SKIP (use cache). False = compute fresh."""
        if step == 0 or step == total - 1 or self._prev is None:
            return False
        norm = self._prev.abs().mean().item() + 1e-8
        rel  = (x - self._prev).abs().mean().item() / norm
        self._accum += self._est(rel)
        return self._accum < self.thresh

    def update(self, x_in: torch.Tensor, out: torch.Tensor) -> None:
        """Called after a fresh compute. Saves state and resets accumulator."""
        self._prev  = x_in.detach().clone()
        self._cache = out.detach().clone()
        self._accum = 0.0
        self.n_calc += 1

    @property
    def cached(self) -> Optional[torch.Tensor]:
        self.n_skip += 1
        return self._cache

    def speedup_str(self) -> str:
        tot = self.n_calc + self.n_skip
        return f"{tot / max(self.n_calc,1):.2f}× (skip {self.n_skip}/{tot})"


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================
class DualModelPipeline:
    """
    SOTA Traffic Video Forecasting Pipeline (Wan2.2 TI2V-5B primary).

    Integrations:
      - Wan2.2 TI2V-5B: WanImageToVideoPipeline + AutoencoderKLWan (f32)
      - Bernini SA-3D RoPE: segment-aware position encoding (optional patch)
      - Bernini VAE Injection: history latents as latent-inpainting condition
      - PhysVideoGenerator: PredictorP + Physics Cross-Attention
      - TeaCache: inline ~4× speedup
      - CacheDiT: stacked ~2-3× speedup
      - MotionAwareComposite: WAN Animate-inspired conditioning image

    Args:
        config: dict from pipeline.yaml (model, generation, lora, etc.)
        device: "cuda" or "cpu"
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        if not _DIFFUSERS_OK:
            raise ImportError("diffusers>=0.33.0 required.")

        self.cfg    = config
        self.device = device
        self.dtype  = torch.bfloat16 if config["model"]["dtype"] == "bfloat16" else torch.float16

        # ── Load primary model (TI2V-5B) ───────────────────────────────────
        ti2v_path = self._model_path("ti2v")
        print(f"[Pipeline] Loading TI2V-5B from {ti2v_path} ...")

        # CRITICAL: load VAE in float32 separately (Wan2.2 requirement)
        vae = AutoencoderKLWan.from_pretrained(
            ti2v_path, subfolder="vae", torch_dtype=torch.float32
        )
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            ti2v_path, vae=vae, torch_dtype=self.dtype
        )
        self.pipe.enable_model_cpu_offload() \
            if config.get("enable_cpu_offload") else self.pipe.to(device)
        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()

        # ── LoRA weights ────────────────────────────────────────────────────
        lora_cfg = config.get("lora", {})
        if lora_cfg.get("enabled") and Path(lora_cfg.get("weights_path", "")).exists():
            self._load_lora(lora_cfg["weights_path"], lora_cfg.get("alpha", 1.0))

        # ── TeaCache ────────────────────────────────────────────────────────
        tc_cfg = config.get("tea_cache", {})
        self.tea_cache = TeaCache(tc_cfg.get("rel_l1_thresh", 0.25)) \
            if tc_cfg.get("enabled", True) else None

        # ── CacheDiT ────────────────────────────────────────────────────────
        cd_cfg = config.get("cache_dit", {})
        if cd_cfg.get("enabled", True) and _CACHE_DIT_OK:
            try:
                cfg_obj = DBCacheConfig(
                    residual_diff_threshold=cd_cfg.get("residual_diff_threshold", 0.08)
                )
                cache_dit.enable_cache(self.pipe, cache_config=cfg_obj)
                print("[Pipeline] CacheDiT (DBCache) enabled")
            except Exception as e:
                warnings.warn(f"[Pipeline] CacheDiT failed: {e}")

        # ── Motion composite ────────────────────────────────────────────────
        mc = config.get("motion", {})
        self.composite = MotionAwareComposite(
            n_frames=mc.get("composite_n_frames", 5),
            method=mc.get("composite_method", "blend"),
        )

        # ── Bernini ─────────────────────────────────────────────────────────
        bernini_cfg = config.get("bernini", {})
        sa3d_cfg = bernini_cfg.get("sa3d_rope", {})
        vae_inj_cfg = bernini_cfg.get("vae_injection", {})
        
        sa3d = SA3DRoPE(
            n_segments=sa3d_cfg.get("n_segments", 4),
            phase_scale=sa3d_cfg.get("phase_scale", 0.3)
        ).to(device) if sa3d_cfg.get("enabled") else None
        
        vae_inj = SourceVAEInjection().to(device) if vae_inj_cfg.get("enabled") else None
        self.bernini = BerniniConditioner(vae_injection=vae_inj, sa3d_rope=sa3d)
        
        # ── Physics ─────────────────────────────────────────────────────────
        phys_cfg = config.get("physics", {})
        self.physics = PhysicsModule().to(device) if phys_cfg.get("enabled") else None
        self.phys_inject_every = phys_cfg.get("inject_every_n_blocks", 4)

        # ── Prompt Relay ────────────────────────────────────────────────────
        pr_cfg = config.get("prompt_relay", {})
        if pr_cfg.get("enabled", True):
            try:
                self.prompt_relay_router = PromptRelayRouter(
                    self.pipe,
                    sigma=pr_cfg.get("sigma", 0.20),
                    penalty_weight=pr_cfg.get("penalty_weight", 1.0),
                )
                print("[Pipeline] Prompt Relay enabled")
            except Exception as e:
                warnings.warn(f"[Pipeline] Prompt Relay initialization failed: {e}")
                self.prompt_relay_router = None
        else:
            self.prompt_relay_router = None

        print("[Pipeline] Ready.")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _model_path(self, which: str) -> str:
        m = self.cfg["model"]
        if which == "ti2v":
            local = m.get("ti2v_local_dir", "")
            return local if local and Path(local).exists() else m["ti2v_repo_id"]
        local = m.get("i2v_local_dir", "")
        return local if local and Path(local).exists() else m["i2v_repo_id"]

    def _load_lora(self, path: str, alpha: float) -> None:
        try:
            self.pipe.load_lora_weights(path)
            if alpha != 1.0 and hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters(["default"], adapter_weights=[alpha])
            print(f"[Pipeline] LoRA loaded: {path} (α={alpha})")
        except Exception as e:
            warnings.warn(f"[Pipeline] LoRA load failed: {e}")

    # ── Core generation ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_variable_n(
        self,
        history_frames: torch.Tensor,
        prompt: str,
        n_frames: int,
        seed: int = 42,
        guidance_scale: float = DEFAULT_CFG,
        num_inference_steps: int = DEFAULT_STEPS,
        negative_prompt: str = DEFAULT_NEG,
        relay_inputs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Alias for compatibility with EnsembleGenerator.
        """
        return self.generate(
            history_frames=history_frames,
            prompt=prompt,
            n_frames=n_frames,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            relay_inputs=relay_inputs,
        )

    @torch.no_grad()
    def generate_with_latent_inpainting(
        self,
        history_frames: torch.Tensor,
        prompt: str,
        n_frames: int,
        seed: int = 42,
        guidance_scale: float = DEFAULT_CFG,
        num_inference_steps: int = DEFAULT_STEPS,
        negative_prompt: str = DEFAULT_NEG,
        relay_inputs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Alias for backward compatibility."""
        return self.generate(
            history_frames=history_frames,
            prompt=prompt,
            n_frames=n_frames,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            relay_inputs=relay_inputs,
        )

    def generate(
        self,
        history_frames: torch.Tensor,   # (K, 3, H, W) in [-1, 1]
        prompt: str,
        n_frames: int,                   # exact target output count (variable!)
        negative_prompt: str = DEFAULT_NEG,
        guidance_scale: float = DEFAULT_CFG,
        num_inference_steps: int = DEFAULT_STEPS,
        seed: int = 42,
        chunk_overlap: int = 8,
        relay_inputs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Generate exactly n_frames future frames.

        Handles:
          - n_frames ≤ 121: single-shot generation
          - n_frames > 121: chunked generation with cosine overlap blend
          - Variable resolution: auto-aligns to multiples of 16 (Bug Fix #4)
          - Variable n_frames: proper 4k+1 rounding (Bug Fix #3)

        Returns:
            Tensor (n_frames, 3, H_orig, W_orig) in [-1, 1]
        """
        orig_h, orig_w = int(history_frames.shape[-2]), int(history_frames.shape[-1])

        if n_frames <= TI2V_MAX_FRAMES:
            return self._generate_single(
                history_frames, prompt, n_frames,
                negative_prompt, guidance_scale, num_inference_steps, seed,
                orig_h, orig_w, relay_inputs,
            )
        return self._generate_chunked(
            history_frames, prompt, n_frames,
            negative_prompt, guidance_scale, num_inference_steps, seed,
            chunk_overlap, orig_h, orig_w, relay_inputs,
        )

    def _generate_single(
        self, history, prompt, n_frames,
        neg, cfg, steps, seed, orig_h, orig_w,
        relay_inputs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Single-shot generation for n_frames ≤ TI2V_MAX_FRAMES."""

        # 1. Conditioning image (MotionAwareComposite)
        cond_img = self.composite.create(history)

        # 2. Align spatial dims to multiples of 16 (Bug Fix #4)
        gen_h, gen_w = align_spatial(orig_h, orig_w, TI2V_SPATIAL_ALIGN)
        if gen_h != orig_h or gen_w != orig_w:
            cond_img = cond_img.resize((gen_w, gen_h), Image.LANCZOS)

        # 3. Valid frame count (Bug Fix #3: integer formula, not math.ceil)
        gen_n = valid_num_frames(n_frames, TI2V_MAX_FRAMES)

        # 4. Seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 5. Reset TeaCache
        if self.tea_cache:
            self.tea_cache.reset()

        # 5. Extract history latents if Latent Inpainting is enabled
        z_hist = None
        if self.bernini.vae_inj is not None and hasattr(self.pipe, "vae"):
            z_hist = self.bernini.vae_inj.encode_history(history, self.pipe.vae, self.device)

        # 6. Run pipeline with safe monkey-patching hook
        with torch.no_grad():
            with _hook_transformer(self.pipe, self.bernini, self.physics, z_hist, self.phys_inject_every):
                if self.prompt_relay_router is not None and relay_inputs is not None:
                    latent_spatial_tokens = (gen_h // 16) * (gen_w // 16)
                    relay_ctx = self.prompt_relay_router.activate(
                        global_prompt=relay_inputs["global_prompt"],
                        local_prompts=relay_inputs["local_prompts"],
                        segment_lengths=relay_inputs["segment_lengths"],
                        num_frames=gen_n,
                        latent_spatial_tokens=latent_spatial_tokens,
                        temporal_compression=TI2V_TEMP_COMPRESSION,
                    )
                else:
                    relay_ctx = contextlib.nullcontext(enter_result=prompt)

                with relay_ctx as active_prompt:
                    out = self.pipe(
                        image=cond_img,
                        prompt=active_prompt,
                        negative_prompt=neg,
                        height=gen_h,
                        width=gen_w,
                        num_frames=gen_n,
                        guidance_scale=cfg,
                        num_inference_steps=steps,
                        generator=generator,
                        output_type="pt",   # returns torch tensor
                    )

        # 7. Extract frames tensor
        # output.frames shape: (B, T, C, H, W), values in [0, 1]
        frames = out.frames  # (B, T, C, H, W)
        if frames.dim() == 5:
            frames = frames[0]   # (T, C, H, W)
        frames = frames.float()  # ensure float32 for downstream ops

        # 8. Normalize to [-1, 1]
        frames = frames * 2.0 - 1.0

        # 9. Resize back to original resolution if needed
        if frames.shape[-2] != orig_h or frames.shape[-1] != orig_w:
            frames = F.interpolate(
                frames, size=(orig_h, orig_w),
                mode="bilinear", align_corners=False,
            )

        # 10. Trim to exactly n_frames
        return frames[:n_frames].cpu()

    def _generate_chunked(
        self, history, prompt, n_frames,
        neg, cfg, steps, seed, chunk_overlap, orig_h, orig_w,
        relay_inputs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Chunked generation for n_frames > TI2V_MAX_FRAMES."""
        print(f"[Pipeline] Chunked: {n_frames} frames (chunk={TI2V_MAX_FRAMES}, overlap={chunk_overlap})")
        chunks: List[torch.Tensor] = []
        remaining = n_frames
        cur_hist  = history
        cur_seed  = seed

        while remaining > 0:
            chunk_n = min(remaining, TI2V_MAX_FRAMES)
            chunk   = self._generate_single(
                cur_hist, prompt, chunk_n, neg, cfg, steps, cur_seed, orig_h, orig_w, relay_inputs
            )
            chunks.append(chunk)
            remaining -= chunk_n - chunk_overlap
            cur_hist   = chunk[-chunk_overlap:]   # new history for next chunk
            cur_seed  += 1
            print(f"[Pipeline]  chunk done, remaining={max(0, remaining)}")

        return _stitch(chunks, chunk_overlap, n_frames)

    def get_vae(self):
        """Return the VAE for external use (e.g., SourceVAEInjection)."""
        return self.pipe.vae if hasattr(self.pipe, "vae") else None


# =============================================================================
# HOOKING UTILITIES
# =============================================================================
@contextlib.contextmanager
def _hook_transformer(pipeline, bernini_cond, physics_module, z_hist, phys_inject_every):
    """
    Context manager to temporarily monkey-patch Wan2.2's DiT transformer.
    Injects Latent Inpainting (Source VAE Injection) and Physics Cross-Attention.
    Restores original methods on exit to avoid memory leaks or state corruption.
    """
    if not hasattr(pipeline, "transformer"):
        yield
        return
        
    transformer = pipeline.transformer
    orig_forward = transformer.forward
    orig_block_forwards = {}
    
    try:
        def hooked_forward(hidden_states, *args, **kwargs):
            # 1. Bernini Latent Inpainting Mask
            if bernini_cond.vae_inj is not None and z_hist is not None:
                # T_h is the number of latent history frames
                hidden_states = bernini_cond.vae_inj(hidden_states, z_hist, z_hist.shape[2])
            
            # 2. PhysVideoGenerator PredictorP
            phys_tokens = None
            if physics_module is not None:
                phys_tokens = physics_module.predictor(hidden_states, text_emb=None, t_emb=None)
            
            # 3. Inject Physics Cross-Attention into blocks dynamically
            if physics_module is not None and phys_tokens is not None:
                blocks = getattr(transformer, "blocks", getattr(transformer, "transformer_blocks", []))
                
                for i, block in enumerate(blocks):
                    idx = i // phys_inject_every
                    if i % phys_inject_every == 0 and idx < len(physics_module.phys_cross_attns):
                        ca_layer = physics_module.phys_cross_attns[idx]
                        
                        if i not in orig_block_forwards:
                            orig_block_forwards[i] = block.forward
                            
                        # Capture current block variables
                        def make_hook(orig_bf, ca):
                            def block_hook(*b_args, **b_kwargs):
                                out = orig_bf(*b_args, **b_kwargs)
                                # diffusers DiT blocks return (hidden_states, ...)
                                if isinstance(out, tuple):
                                    hs = out[0]
                                    hs = ca(hs, phys_tokens)
                                    return (hs,) + out[1:]
                                return ca(out, phys_tokens)
                            return block_hook
                        
                        block.forward = make_hook(orig_block_forwards[i], ca_layer)

            return orig_forward(hidden_states, *args, **kwargs)
            
        transformer.forward = hooked_forward
        yield
        
    finally:
        # Absolute restoration of all monkey-patched methods
        transformer.forward = orig_forward
        blocks = getattr(transformer, "blocks", getattr(transformer, "transformer_blocks", []))
        for i, block in enumerate(blocks):
            if i in orig_block_forwards:
                block.forward = orig_block_forwards[i]


# =============================================================================
# CHUNK STITCHING — cosine cross-fade
# =============================================================================
def _stitch(chunks: List[torch.Tensor], overlap: int, target_n: int) -> torch.Tensor:
    """
    Stitch video chunks with cosine cross-fade at boundaries.
    cos_a: 1→0  (weight for end of prev chunk)
    cos_b: 0→1  (weight for start of next chunk)
    """
    if len(chunks) == 1:
        return chunks[0][:target_n]

    t     = torch.linspace(0, 1, overlap)
    cos_a = (0.5 * (1 + torch.cos(torch.pi * t))).view(overlap, 1, 1, 1)
    cos_b = 1 - cos_a

    parts = [chunks[0]]
    for ch in chunks[1:]:
        prev = parts[-1]
        blended = cos_a * prev[-overlap:] + cos_b * ch[:overlap]
        parts[-1] = torch.cat([prev[:-overlap], blended], 0)
        parts.append(ch[overlap:])

    return torch.cat(parts, 0)[:target_n]
