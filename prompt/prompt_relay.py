"""
src/prompt/prompt_relay.py
===========================
Prompt Relay – Inference-Time Temporal Control for Multi-Event Video Generation

Reference
---------
Paper:  "Prompt Relay: Inference-Time Temporal Control for Multi-Event
         Video Generation" – Chen, Huang, Liu (NTU S-Lab, 2026)
arXiv:  https://arxiv.org/abs/2604.10030
Code:   https://github.com/GordonChen19/Prompt-Relay
Status: Officially integrated into Wan2.2 (https://github.com/Wan-Video/Wan2.2)

Core Idea
---------
Video diffusion models suffer from "temporal entanglement": when a long
paragraph-style prompt describes multiple sequential events, the semantics
from later events bleed into early frames and vice versa.

For WTS specifically:
  - caption_pedestrian describes pedestrian behavior
  - caption_vehicle describes vehicle behavior
  - These often refer to DIFFERENT temporal windows of the same segment

Prompt Relay fixes this by routing each local_prompt to its designated
temporal window via a distance-based penalty in cross-attention:

    A_modified(Q, K) = A(Q, K) - λ · C(Q, K)

where C(Q, K) is a Gaussian penalty that suppresses attention between
queries in segment i and key tokens belonging to a different segment j:

    C(q, k)_ij = exp( - ||pos(q) - center(i)||² / (2σ²) )
                 × 1[k ∈ segment_j, j ≠ i]

Key Properties
--------------
- Training-free: no architectural changes, no retraining needed
- Plug-and-play: injected as a hook into cross-attention layers
- No extra compute: penalty is added to attention logits pre-softmax
- Works with Wan2.2 DiT attention layers

Implementation Strategy
-----------------------
We implement this as forward hooks on the DiT's cross-attention modules.
For Wan2.2 (Diffusers implementation), the attention layers follow the
standard Diffusers `Attention` interface.

The hook modifies the raw attention weight matrix BEFORE the softmax,
adding the penalty to suppress out-of-segment token attention.

How Segment Boundaries are Set
--------------------------------
From CoECT:
  segment_lengths: [0.3, 0.4, 0.3]  # fractions of total video length
  local_prompts: ["event 1 desc", "event 2 desc", "event 3 desc"]

We convert to frame indices:
  if num_frames=33, lengths=[0.3, 0.4, 0.3]:
    segment 0: frames 0-9
    segment 1: frames 10-23
    segment 2: frames 24-32

Each token in the latent space corresponds to a temporal position.
The penalty ensures segment 1's attention keys do not influence segment 0's queries.
"""

import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Segment Boundary Computation
# ---------------------------------------------------------------------------

def compute_segment_boundaries(
    num_frames: int,
    segment_lengths: List[float],
    temporal_compression: int = 4,
) -> List[Tuple[int, int]]:
    """
    Convert fractional segment lengths to (start, end) frame index pairs.
    """
    # Standard formula for Wan2.2 VAE
    latent_frames = ((num_frames - 1) // temporal_compression) + 1
    
    total = sum(segment_lengths)
    if total == 0:
        segment_lengths = [1.0 / len(segment_lengths)] * len(segment_lengths)
        total = 1.0
    normalized = [s / total for s in segment_lengths]
    
    boundaries = []
    current_frame = 0
    cumulative_frac = 0.0
    
    for i, frac in enumerate(normalized):
        start = current_frame
        cumulative_frac += frac
        
        if i == len(normalized) - 1:
            end = latent_frames - 1
        else:
            # Calculate end frame based on cumulative sum to spread rounding errors
            target_cumulative_frame = round(cumulative_frac * latent_frames)
            end = max(start, target_cumulative_frame - 1)
            
        boundaries.append((start, end))
        current_frame = end + 1
        
    return boundaries


# ---------------------------------------------------------------------------
# Prompt Relay Attention Hook
# ---------------------------------------------------------------------------

class PromptRelayHook:
    """
    Forward hook that injects the Prompt Relay temporal penalty into a
    single cross-attention layer.
    
    The hook is registered on an `nn.Module` and modifies the attention
    weights BEFORE softmax via `register_forward_pre_hook` or by wrapping
    the attention computation.
    
    Implementation Details
    ----------------------
    Wan2.2 uses `diffusers.models.attention_processor.AttnProcessor2_0`
    or a custom Wan attention processor. We wrap the processor's `__call__`
    to intercept the attention logits.
    
    The modified attention logits matrix has shape:
        (batch × heads, seq_len_query, seq_len_key)
    
    Where:
        seq_len_query = spatial_tokens × temporal_frames (video latent tokens)
        seq_len_key   = text_token_length (from T5 encoder)
    
    The penalty applies at the (query, key) level:
      - For each query token at temporal position t:
        → Find which segment i contains frame t
        → For each key token k_j from local prompt j ≠ i:
          → penalty = exp(-(t_frac - center_i)^2 / (2σ^2)) × weight_mask[k_j]
    
    Args:
        segment_boundaries: list of (start, end) frame idx pairs (latent space)
        local_prompt_token_ranges: list of (start, end) key-space token idx
                                   for each local prompt
        sigma: Gaussian decay parameter (controls boundary sharpness)
               Lower σ → harder boundary; paper suggests σ ≈ 0.15-0.25
        latent_spatial_tokens: number of spatial tokens per latent frame
                               = (H // 8) * (W // 8) // (patch_size^2)
    """
    
    def __init__(
        self,
        segment_boundaries: List[Tuple[int, int]],
        local_prompt_token_ranges: List[Tuple[int, int]],
        sigma: float = 0.2,
        latent_spatial_tokens: int = 1440,  # default for 720p with patch=2
        penalty_weight: float = 1.0,
    ):
        self.segment_boundaries = segment_boundaries
        self.local_prompt_token_ranges = local_prompt_token_ranges
        self.sigma = sigma
        self.latent_spatial_tokens = latent_spatial_tokens
        self.penalty_weight = penalty_weight
        
        # Number of segments
        self.n_segments = len(segment_boundaries)
        
        # Total latent frames (inferred from boundaries)
        self.n_latent_frames = segment_boundaries[-1][1] + 1 if segment_boundaries else 1
        
        # Precompute: for each latent frame, which segment it belongs to
        self._frame_to_segment = self._build_frame_segment_map()
        
        # Precompute penalty matrices (will be recomputed if shapes differ)
        self._cached_penalty: Optional[torch.Tensor] = None
        self._cached_penalty_shape: Optional[Tuple] = None
    
    def _build_frame_segment_map(self) -> Dict[int, int]:
        """Map each latent frame index to its segment index."""
        mapping = {}
        for seg_idx, (start, end) in enumerate(self.segment_boundaries):
            for f in range(start, end + 1):
                mapping[f] = seg_idx
        return mapping
    
    def _get_frame_for_query_token(self, query_token_idx: int) -> int:
        """
        Get the temporal frame index for a given query token position.
        
        In Wan2.2's 3D VAE, video tokens are flattened as:
            [frame_0_spatial_tokens, frame_1_spatial_tokens, ..., frame_T_spatial_tokens]
        
        So token i → frame i // spatial_tokens_per_frame
        """
        return query_token_idx // self.latent_spatial_tokens
    
    def apply_penalty(
        self,
        attn_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the Prompt Relay penalty to attention logits.
        """
        batch_heads, Q, K = attn_logits.shape
        device = attn_logits.device
        dtype = attn_logits.dtype
        
        # ── VRAM SAFETY LOCK ──────────────────────────────────────────
        # If K == Q or K > 2048, this is likely Self-Attention. Do NOT apply penalty.
        if K == Q or K > 2048:
            return attn_logits
            
        # Check if Q length is valid
        if Q > self.n_latent_frames * self.latent_spatial_tokens:
            return attn_logits
            
        if len(self.local_prompt_token_ranges) == 0:
            return attn_logits
            
        penalty = self._compute_penalty_vectorized(Q, K, device, dtype)
        attn_logits = attn_logits - penalty.unsqueeze(0)
        return attn_logits

    def _compute_penalty_vectorized(
        self, query_len: int, key_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        shape = (query_len, key_len, device, dtype)
        if self._cached_penalty_shape == shape:
            return self._cached_penalty
            
        # 1. Compute T_Norm for all Queries (Q, 1)
        t_q = torch.arange(query_len, device=device, dtype=torch.float32)
        t_q_normalized = (t_q // self.latent_spatial_tokens) / max(self.n_latent_frames - 1, 1)
        t_norm = t_q_normalized.unsqueeze(1) # (Q, 1)
        
        # 2. Build Mapping Frame -> Segment (Q, 1)
        frame_indices = (t_q // self.latent_spatial_tokens).long()
        frame_to_seg_tensor = torch.zeros(self.n_latent_frames + 1, dtype=torch.long, device=device)
        for f, s in self._frame_to_segment.items():
            if f < len(frame_to_seg_tensor):
                frame_to_seg_tensor[f] = s
        seg_q = frame_to_seg_tensor[frame_indices].unsqueeze(1) # (Q, 1)

        # 3. Initialize Penalty matrix (Q, K)
        penalty = torch.zeros((query_len, key_len), device=device, dtype=torch.float32)
        
        # 4. Vectorized computation per Segment
        for seg_j, (start_k, end_k) in enumerate(self.local_prompt_token_ranges):
            # Mask for Key (1, K)
            k_mask = torch.zeros(key_len, device=device, dtype=torch.float32)
            if start_k < key_len:
                k_mask[start_k:min(end_k + 1, key_len)] = 1.0
            k_mask = k_mask.unsqueeze(0)
            
            # Center of segment j
            start_f, end_f = self.segment_boundaries[seg_j]
            center_j = (start_f + end_f) / 2.0 / max(self.n_latent_frames - 1, 1)
            
            # Gaussian for ALL queries (Q, 1)
            gauss = torch.exp(-((t_norm - center_j) ** 2) / (2 * self.sigma ** 2))
            
            # Only apply if Query does NOT belong to Segment j (Q, 1)
            valid_q_mask = (seg_q != seg_j).float()
            
            # Broadcasting: (Q, 1) * (Q, 1) * (1, K) -> (Q, K)
            penalty += gauss * valid_q_mask * k_mask
            
        penalty = penalty * self.penalty_weight
        
        self._cached_penalty = penalty.to(dtype=dtype)
        self._cached_penalty_shape = shape
        return self._cached_penalty


# ---------------------------------------------------------------------------
# Prompt Relay Router – High-Level Interface
# ---------------------------------------------------------------------------

class PromptRelayRouter:
    """
    High-level Prompt Relay manager for Wan2.2 video generation.
    
    Coordinates:
      1. Text encoding: global_prompt + local_prompts → encoder embeddings
      2. Token range tracking: knowing which encoder output tokens belong
         to which local prompt
      3. Hook injection: registers PromptRelayHook on all DiT cross-attention layers
      4. Hook cleanup: removes all hooks after generation
    
    Usage (within inference pipeline):
        >>> router = PromptRelayRouter(pipe, sigma=0.2)
        >>> with router.activate(
        ...     global_prompt="overhead traffic scene...",
        ...     local_prompts=["vehicle braking hard", "pedestrian stepping back"],
        ...     segment_lengths=[0.4, 0.6],
        ...     num_frames=33,
        ... ):
        ...     output = pipe(prompt=router.combined_prompt, ...)
    
    Args:
        pipe: Wan2.2 DiffusionPipeline (diffusers)
        sigma: Gaussian decay for cross-segment penalty
        penalty_weight: scale factor for penalty strength
    """
    
    def __init__(
        self,
        pipe: Any,  # WanImageToVideoPipeline
        sigma: float = 0.2,
        penalty_weight: float = 1.0,
    ):
        self.pipe = pipe
        self.sigma = sigma
        self.penalty_weight = penalty_weight
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self.combined_prompt: str = ""
        self._active = False
        
        # ORIGINAL FORWARD STORAGE: Essential to survive __exit__ and prevent circular leaks
        self._original_forwards: Dict[str, Callable] = {}
    
    def _encode_text(
        self,
        text: str,
    ) -> torch.Tensor:
        """
        Encode text through the pipeline's text encoder (T5).
        Returns encoder hidden states (1, seq_len, hidden_dim).
        """
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder
        device = next(text_encoder.parameters()).device
        dtype = next(text_encoder.parameters()).dtype
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).to(device)
        
        with torch.no_grad():
            embeddings = text_encoder(**inputs).last_hidden_state
        
        return embeddings
    
    def build_combined_prompt(
        self,
        global_prompt: str,
        local_prompts: List[str],
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Build the combined text prompt fed to the pipeline AND compute
        the token range for each local prompt within the combined string.
        
        Strategy:
          combined = global_prompt + " [SEP] " + local_prompts[0] + " [SEP] " + ...
        
        Then encode the combined prompt and identify token indices that
        correspond to each local prompt's tokens.
        
        Returns:
            combined_prompt: str
            token_ranges: list of (start_token, end_token) per local prompt
        """
        sep = " . "
        segments = [global_prompt] + local_prompts
        combined = sep.join(segments)
        
        # Tokenize to find boundaries
        # Note: actual token ranges depend on the T5 tokenizer
        tokenizer = self.pipe.tokenizer if hasattr(self.pipe, 'tokenizer') else None
        
        token_ranges = []
        if tokenizer is not None:
            # Encode each component and track cumulative lengths
            global_tokens = tokenizer(global_prompt, add_special_tokens=False).input_ids
            sep_tokens = tokenizer(sep, add_special_tokens=False).input_ids
            
            cursor = len(global_tokens) + len(sep_tokens)  # skip global + first sep
            
            for i, lp in enumerate(local_prompts):
                lp_tokens = tokenizer(lp, add_special_tokens=False).input_ids
                token_ranges.append((cursor, cursor + len(lp_tokens) - 1))
                cursor += len(lp_tokens) + len(sep_tokens)
        else:
            # Fallback: equal token splits (rough approximation)
            total_est = 256  # typical T5 max length
            per_seg = total_est // (len(local_prompts) + 1)
            start = per_seg  # skip global
            for lp in local_prompts:
                token_ranges.append((start, start + per_seg - 1))
                start += per_seg
        
        return combined, token_ranges
    
    def _inject_hooks(
        self,
        hook: PromptRelayHook,
    ) -> None:
        """
        Register the Prompt Relay penalty hook on ALL cross-attention layers
        in the DiT (denoising transformer).
        
        In Wan2.2 (Diffusers), cross-attention layers are identified by having
        both self-attention and cross-attention; we target cross-attention only.
        
        The hook wraps the attention processor's __call__ to intercept logits.
        """
        transformer = self.pipe.transformer if hasattr(self.pipe, 'transformer') else None
        if transformer is None:
            warnings.warn("[PromptRelay] No transformer found; hooks not injected.")
            return
        
        def make_hook(hook_obj: PromptRelayHook):
            """Factory to create a forward hook that applies penalty."""
            def forward_pre_hook(module, args):
                # args structure depends on Diffusers version
                # We'll patch at a higher level instead (see _patch_attention below)
                return args
            return forward_pre_hook
        
        # Alternative: patch the attention forward method directly
        self._patch_attention_layers(transformer, hook)
    
    def _patch_attention_layers(
        self,
        transformer: nn.Module,
        hook: PromptRelayHook,
    ) -> None:
        """Patch safely with original state storage."""
        patched_count = 0
        
        for name, module in transformer.named_modules():
            is_cross_attn = (
                "cross_attn" in name.lower()
                or "attn2" in name.lower()
                or ("attn" in name.lower() and hasattr(module, "to_k"))
            )
            if not is_cross_attn: continue
            if not hasattr(module, "to_q") or not hasattr(module, "to_k"): continue

            # PREVENT DOUBLE-PATCHING
            if name in self._original_forwards:
                continue 
                
            # Store original function
            self._original_forwards[name] = module.forward
            original_forward = module.forward
            
            # Use Closure wrapper to avoid Late Binding and Reference Cycles
            def make_patched_forward(orig_fn, hook_obj, mod):
                def patched_forward(
                    hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=None,
                    **kwargs,
                ):
                    if encoder_hidden_states is None:
                        return orig_fn(hidden_states, None, attention_mask, **kwargs)
                    
                    q = mod.to_q(hidden_states)
                    k = mod.to_k(encoder_hidden_states)
                    v = mod.to_v(encoder_hidden_states)
                    
                    head_dim = q.shape[-1] // mod.heads if hasattr(mod, 'heads') else q.shape[-1]
                    scale = head_dim ** -0.5
                    
                    B, Q_len, _ = q.shape
                    _, K_len, _ = k.shape
                    n_heads = mod.heads if hasattr(mod, 'heads') else 1
                    
                    q = q.reshape(B, Q_len, n_heads, head_dim).transpose(1, 2).reshape(B * n_heads, Q_len, head_dim)
                    k = k.reshape(B, K_len, n_heads, head_dim).transpose(1, 2).reshape(B * n_heads, K_len, head_dim)
                    v = v.reshape(B, K_len, n_heads, head_dim).transpose(1, 2).reshape(B * n_heads, K_len, head_dim)
                    
                    attn_logits = torch.bmm(q, k.transpose(1, 2)) * scale  # (B*H, Q, K)
                    
                    # Apply penalty
                    attn_logits = hook_obj.apply_penalty(attn_logits)
                    
                    if attention_mask is not None:
                        attn_logits = attn_logits + attention_mask
                    
                    attn_weights = F.softmax(attn_logits, dim=-1)
                    out = torch.bmm(attn_weights, v)  # (B*H, Q, head_dim)
                    
                    out = out.reshape(B, n_heads, Q_len, head_dim).transpose(1, 2).reshape(B, Q_len, n_heads * head_dim)
                    
                    out = mod.to_out[0](out) if hasattr(mod, 'to_out') else out
                    if hasattr(mod, 'to_out') and len(mod.to_out) > 1:
                        out = mod.to_out[1](out)
                    
                    return out
                return patched_forward
            
            module.forward = make_patched_forward(original_forward, hook, module)
            patched_count += 1
            
        print(f"[PromptRelay] Activated: Patched {patched_count} cross-attention layers")
    
    def _restore_attention_layers(self, transformer: nn.Module) -> None:
        """Remove all patches and clean up garbage collection."""
        restored_count = 0
        for name, module in transformer.named_modules():
            if name in self._original_forwards:
                module.forward = self._original_forwards[name]
                restored_count += 1
                
        self._original_forwards.clear()
        print(f"[PromptRelay] Deactivated: Restored {restored_count} original layers")
    
    def activate(
        self,
        global_prompt: str,
        local_prompts: List[str],
        segment_lengths: List[float],
        num_frames: int,
        latent_spatial_tokens: int = 1440,
        temporal_compression: int = 4,
    ):
        """
        Context manager: activates Prompt Relay hooks for the duration of
        a single generation call.
        
        Args:
            global_prompt: scene-wide context (persistent characters, location)
            local_prompts: event-specific descriptions, one per temporal segment
            segment_lengths: fractional lengths for each local prompt (sum ~1.0)
            num_frames: total number of frames to generate
            latent_spatial_tokens: spatial tokens per latent frame
            temporal_compression: VAE temporal downsampling factor
        
        Usage:
            >>> with router.activate(...) as combined_prompt:
            ...     output = pipe(prompt=combined_prompt, ...)
        """
        return _PromptRelayContext(
            router=self,
            global_prompt=global_prompt,
            local_prompts=local_prompts,
            segment_lengths=segment_lengths,
            num_frames=num_frames,
            latent_spatial_tokens=latent_spatial_tokens,
            temporal_compression=temporal_compression,
        )


class _PromptRelayContext:
    """
    Context manager returned by PromptRelayRouter.activate().
    Handles hook injection and cleanup.
    """
    
    def __init__(self, router: PromptRelayRouter, **kwargs):
        self.router = router
        self.kwargs = kwargs
        self.combined_prompt = ""
    
    def __enter__(self) -> str:
        """
        Set up Prompt Relay hooks.
        
        Returns the combined prompt string that should be passed to the
        pipeline's `prompt` argument.
        """
        router = self.router
        kw = self.kwargs
        
        # Build combined prompt + compute token ranges
        combined, token_ranges = router.build_combined_prompt(
            kw["global_prompt"], kw["local_prompts"]
        )
        self.combined_prompt = combined
        router.combined_prompt = combined
        
        # Compute segment boundaries in latent frame space
        segment_boundaries = compute_segment_boundaries(
            num_frames=kw["num_frames"],
            segment_lengths=kw["segment_lengths"],
            temporal_compression=kw["temporal_compression"],
        )
        
        # Build hook
        relay_hook = PromptRelayHook(
            segment_boundaries=segment_boundaries,
            local_prompt_token_ranges=token_ranges,
            sigma=router.sigma,
            latent_spatial_tokens=kw["latent_spatial_tokens"],
            penalty_weight=router.penalty_weight,
        )
        
        # Inject into model
        transformer = router.pipe.transformer if hasattr(router.pipe, 'transformer') else None
        if transformer is not None:
            router._patch_attention_layers(transformer, relay_hook)
        
        return combined
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Remove all Prompt Relay hooks to restore normal behavior."""
        transformer = self.router.pipe.transformer if hasattr(self.router.pipe, 'transformer') else None
        if transformer is not None:
            self.router._restore_attention_layers(transformer)
