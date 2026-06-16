"""
src/inference/ensemble.py
==========================
Multi-Candidate Ensemble Generator & Selector

Why Ensemble?
─────────────
Video generation is stochastic. Different seeds → different plausible futures.
By generating K candidates (K=3-5) and selecting the best:
  1. CLIP-S improves: at least one candidate will align better with text
  2. PSNR/SSIM improve: candidate closest to GT trajectory selected
  3. FVD improves: most temporally coherent video selected

TeaCache + CacheDiT give us 4-9× speedup → K candidates ≈ 1 baseline cost.

Scoring Strategy
────────────────
Each candidate is scored on:
  1. CLIP-S: similarity between combined caption and generated frames
     (measures text-video alignment)
  2. History Consistency: last history frame ↔ first generated frame similarity
     (measures temporal continuity from conditioning)
  3. Temporal Coherence: frame-to-frame SSIM within generated sequence
     (measures internal video quality, proxy for FVD)

Winner = argmax(weighted_composite_score)

V-JEPA 2.1 Integration (Optional)
─────────────────────────────────
Paper: arXiv 2603.14482 – "V-JEPA 2.1: Unlocking Dense Features in
       Video Self-Supervised Learning" (Meta, 2026)

V-JEPA 2.1 produces dense, spatially-grounded video representations.
We use it as an additional scoring signal:
  V-JEPA score = cosine_similarity(
      vjepa(history_frames),
      vjepa(generated_frames[:K_overlap])  # first K frames of generated
  )
High V-JEPA similarity → generated video is perceptually consistent
with the history (scene identity, object appearance preserved).

This is analogous to how WAN Animate uses reference image similarity
to ensure character consistency across video frames.

References:
    V-JEPA 2.1:   arXiv 2603.14482
    WAN Animate:  https://humanaigc.github.io/wan-animate/
    CLIP:         openai/clip-vit-large-patch14
    TeaCache:     arXiv 2411.19108 (speedup enabling this ensemble)
"""

from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# CLIP for CLIP-S scoring
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    warnings.warn("open_clip not installed; CLIP-S scoring disabled.")


# ═══════════════════════════════════════════════════════════════════
# CLIP-S SCORER
# ═══════════════════════════════════════════════════════════════════

class CLIPScorer:
    """
    Compute CLIP-S (text-video alignment) score.

    CLIP-S measures how well generated video frames align with the
    text description. Higher = better text-video alignment.

    Implementation:
        For each candidate video:
            1. Subsample K frames uniformly from the video
            2. Encode each frame with CLIP image encoder
            3. Encode the text prompt with CLIP text encoder
            4. CLIP-S = mean cosine similarity(frame_embed, text_embed)

    Args:
        model_name: CLIP model to use
        device: compute device
        n_frames_to_score: how many frames to subsample for efficiency
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda",
        n_frames_to_score: int = 8,
    ):
        self.device = device
        self.n_frames = n_frames_to_score

        if not CLIP_AVAILABLE:
            self.model = None
            self.preprocess = None
            warnings.warn("[CLIPScorer] open_clip not available; returning 0.0")
            return

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        print(f"[CLIPScorer] Loaded {model_name} on {device}")

    @torch.no_grad()
    def score(
        self,
        frames: torch.Tensor,    # (N, 3, H, W) in [-1, 1]
        prompt: str,
    ) -> float:
        """
        Compute CLIP-S for a video candidate.

        Args:
            frames: video frames (N, 3, H, W) normalized to [-1, 1]
            prompt: text description

        Returns:
            float: mean cosine similarity in [0, 1]
        """
        if self.model is None:
            return 0.0

        # Subsample frames uniformly
        N = frames.shape[0]
        indices = np.linspace(0, N - 1, self.n_frames, dtype=int)
        sampled = frames[indices]  # (n_frames, 3, H, W)

        # Convert to PIL and preprocess
        clip_images = []
        for i in range(sampled.shape[0]):
            frame = sampled[i]
            # [-1,1] → [0,255] PIL
            arr = ((frame.cpu().float() + 1.0) / 2.0).clamp(0, 1)
            arr = (arr * 255).byte().permute(1, 2, 0).numpy()
            pil = Image.fromarray(arr, mode="RGB")
            clip_images.append(self.preprocess(pil))

        image_batch = torch.stack(clip_images, dim=0).to(self.device)
        text_batch = self.tokenizer([prompt]).to(self.device)

        # Encode
        image_features = self.model.encode_image(image_batch)
        text_features = self.model.encode_text(text_batch)

        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Cosine similarities for each frame
        sims = (image_features @ text_features.T).squeeze(-1)  # (n_frames,)
        return float(sims.mean().item())


# ═══════════════════════════════════════════════════════════════════
# HISTORY CONSISTENCY SCORER
# ═══════════════════════════════════════════════════════════════════

class HistoryConsistencyScorer:
    """
    Score temporal continuity between history and generated frames.

    Measures how smoothly the generated video continues from the
    last history frame. A large visual jump = bad score.

    Two metrics:
      1. Pixel L2 distance: last_hist_frame vs first_gen_frame (lower = better)
      2. SSIM: structural similarity at the boundary

    Returns:
        score in [0, 1] where 1 = perfect continuity
    """

    def score(
        self,
        history_frames: torch.Tensor,   # (K, 3, H, W) in [-1, 1]
        generated_frames: torch.Tensor, # (N, 3, H, W) in [-1, 1]
    ) -> float:
        """Compute history→generated boundary continuity score using SSIM proxy."""
        last_hist = history_frames[-1].float()   # (3, H, W)
        first_gen = generated_frames[0].float()  # (3, H, W)

        # Ensure same device
        first_gen = first_gen.to(last_hist.device)

        # Simple SSIM proxy (Structural Consistency) avoids VAE color shift penalties
        mu1 = last_hist.mean()
        mu2 = first_gen.mean()
        sigma1 = last_hist.var()
        sigma2 = first_gen.var()
        sigma12 = ((last_hist - mu1) * (first_gen - mu2)).mean()

        C1, C2 = 0.01**2, 0.03**2
        ssim = (
            (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
            / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
        ).item()

        return max(0.0, min(1.0, ssim))


# ═══════════════════════════════════════════════════════════════════
# TEMPORAL COHERENCE SCORER
# ═══════════════════════════════════════════════════════════════════

class TemporalCoherenceScorer:
    """
    Score internal temporal coherence of generated video.

    Measures frame-to-frame visual similarity WITHIN the generated
    video. Very high similarity = too static; too low = too jerky.
    Ideal: moderate similarity reflecting realistic motion.

    For traffic: frames change slowly (overhead camera), so target
    similarity should be high (≥ 0.85 SSIM between consecutive frames).

    Returns:
        score in [0, 1]
    """

    def score(
        self,
        frames: torch.Tensor,  # (N, 3, H, W) in [-1, 1]
        sample_stride: int = 3,
        anomaly_threshold: float = 0.65,
    ) -> float:
        """
        Compute Temporal Coherence via Anomaly Detection.
        Instead of averaging (which rewards stationary video), 
        we count the ratio of broken frames (SSIM < threshold).

        Args:
            frames: generated video frames
            sample_stride: compute SSIM every `stride` frames for speed
            anomaly_threshold: SSIM drop below this indicates a "broken" frame
        """
        N = frames.shape[0]
        if N < 2:
            return 1.0

        ssim_values = []
        for i in range(0, N - 1, sample_stride):
            f1 = frames[i].float()    # (3, H, W)
            f2 = frames[i + 1].float()

            # Simple SSIM approximation via correlation
            mu1 = f1.mean()
            mu2 = f2.mean()
            sigma1 = f1.var()
            sigma2 = f2.var()
            sigma12 = ((f1 - mu1) * (f2 - mu2)).mean()

            C1, C2 = 0.01**2, 0.03**2
            ssim = (
                (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
                / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
            ).item()

            ssim_values.append(ssim)

        if not ssim_values:
            return 1.0

        ssim_array = np.array(ssim_values)
        broken_frames_ratio = np.mean(ssim_array < anomaly_threshold)

        # 1.0 = perfect coherence (no breaks), 0.0 = completely broken
        return float(1.0 - broken_frames_ratio)


# ═══════════════════════════════════════════════════════════════════
# V-JEPA 2.1 SCORER (Optional)
# ═══════════════════════════════════════════════════════════════════

class VJEPAScorer:
    """
    Dense video feature scorer using V-JEPA 2.1.

    Paper: "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised
            Learning" – Mur-Labadia et al. (Meta, 2026)
    arXiv: https://arxiv.org/abs/2603.14482

    V-JEPA 2.1 learns dense, spatially-grounded video representations
    via a masking-based self-supervised objective with hierarchical
    self-supervision across multiple encoder layers.

    We use it to score perceptual consistency between the history frames
    and the beginning of each generated candidate:

        score = cosine_similarity(
            vjepa(history_frames[-overlap:]),
            vjepa(generated_frames[:overlap])
        )

    This measures whether the generated video "looks like" a natural
    continuation of the history in the deep perceptual feature space.

    High V-JEPA score → better FVD (they measure similar things).

    Args:
        model_name: V-JEPA 2.1 model on HuggingFace
        device: compute device
        overlap_frames: how many frames to compare at boundary
    """

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc16-256",
        device: str = "cuda",
        overlap_frames: int = 4,
    ):
        self.device = device
        self.overlap = overlap_frames
        self.model = None

        try:
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained(model_name).to(device).eval()
            self.processor = AutoProcessor.from_pretrained(model_name)
            print(f"[VJEPAScorer] Loaded {model_name} on {device}")
        except Exception as e:
            warnings.warn(f"[VJEPAScorer] V-JEPA load failed: {e}. Scoring disabled.")

    @torch.no_grad()
    def score(
        self,
        history_frames: torch.Tensor,    # (K, 3, H, W) in [-1, 1]
        generated_frames: torch.Tensor,  # (N, 3, H, W) in [-1, 1]
    ) -> float:
        """Compute V-JEPA perceptual continuity score."""
        if self.model is None:
            return 0.5  # neutral score when unavailable

        # Use last [overlap] history frames and first [overlap] generated
        hist_clip = history_frames[-self.overlap:]   # (overlap, 3, H, W)
        gen_clip = generated_frames[:self.overlap]   # (overlap, 3, H, W)

        def encode(clip: torch.Tensor) -> torch.Tensor:
            """Encode a video clip with V-JEPA."""
            # [-1, 1] → [0, 1]
            clip_01 = ((clip.float() + 1.0) / 2.0).clamp(0, 1)
            # Add batch dimension: (1, C, T, H, W)
            clip_input = clip_01.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

            # Resize to 256×256 (V-JEPA input size)
            if clip_input.shape[-2:] != (256, 256):
                clip_input = F.interpolate(
                    clip_input.view(-1, 3, *clip_input.shape[-2:]),
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                ).view(*clip_input.shape[:3], 256, 256)

            features = self.model(clip_input).last_hidden_state
            return features.mean(dim=[1, 2])  # (1, hidden_dim) global pool

        hist_feat = encode(hist_clip)
        gen_feat = encode(gen_clip)

        # Cosine similarity
        sim = F.cosine_similarity(hist_feat, gen_feat, dim=-1)
        return float(sim.item())


# ═══════════════════════════════════════════════════════════════════
# ENSEMBLE GENERATOR & SELECTOR
# ═══════════════════════════════════════════════════════════════════

class EnsembleGenerator:
    """
    Multi-candidate video ensemble generator and selector.

    Generates K candidates with different seeds, scores each candidate,
    and returns the best-scoring one.

    Why K candidates work:
        Generation is stochastic (Gaussian noise → video).
        Each seed explores a different region of the posterior P(video|text,image).
        The "correct" future is in this posterior, so sampling K times and
        picking the best is provably better than any single sample.

    Scoring:
        composite_score = w1 × CLIP-S + w2 × HistConsistency + w3 × TemporalCoherence
        + w4 × V-JEPA (if available)

    The weights balance text-alignment vs visual quality vs temporal smoothness.

    Args:
        pipeline: DualModelPipeline instance
        n_candidates: number of candidates to generate (K)
        base_seed: seeds will be [base_seed, base_seed+1, ..., base_seed+K-1]
        clip_weight: weight for CLIP-S score
        hist_weight: weight for history consistency score
        temp_weight: weight for temporal coherence score
        vjepa_weight: weight for V-JEPA score (0 to disable)
        enable_vjepa: whether to initialize V-JEPA scorer
        device: compute device
    """

    def __init__(
        self,
        pipeline,  # DualModelPipeline
        n_candidates: int = 5,
        base_seed: int = 42,
        clip_weight: float = 0.40,
        hist_weight: float = 0.35,
        temp_weight: float = 0.15,
        vjepa_weight: float = 0.10,
        enable_vjepa: bool = True,
        clip_device: str = "cuda",
        vjepa_device: str = "cuda",
    ):
        self.pipeline = pipeline
        self.n_candidates = n_candidates
        self.base_seed = base_seed
        self.weights = {
            "clip_s": clip_weight,
            "hist": hist_weight,
            "temp": temp_weight,
            "vjepa": vjepa_weight,
        }

        # Initialize scorers
        self.clip_scorer = CLIPScorer(device=clip_device) if CLIP_AVAILABLE else None
        self.hist_scorer = HistoryConsistencyScorer()
        self.temp_scorer = TemporalCoherenceScorer()
        self.vjepa_scorer = VJEPAScorer(device=vjepa_device) if enable_vjepa else None

    def generate_and_select(
        self,
        history_frames: torch.Tensor,     # (K, 3, H, W)
        prompt: str,
        n_frames: int,
        generation_kwargs: Dict[str, Any],
        prompt_package: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate K candidates and return the best one.

        Args:
            history_frames: conditioning history
            prompt: combined text prompt
            n_frames: number of frames to generate
            generation_kwargs: additional kwargs for generate_variable_n()
                (guidance_scale, num_inference_steps, etc.)
            prompt_package: full prompt package from PromptBuilder
                            (used to get enhanced_ped/veh for CLIP scoring)

        Returns:
            Tuple:
                best_frames: (n_frames, 3, H, W) best candidate
                info: dict with scores and metadata for each candidate
        """
        candidates = []
        scores = []

        # Get the caption for CLIP-S scoring
        clip_text = prompt
        if prompt_package:
            # Use a more specific prompt that combines ped+veh for CLIP
            ped = prompt_package.get("enhanced_ped_prompt", "")
            veh = prompt_package.get("enhanced_veh_prompt", "")
            clip_text = f"{ped} {veh}"[:500]  # clip prompt to reasonable length

        relay_inputs = prompt_package.get("relay_inputs") if prompt_package else None

        for i in range(self.n_candidates):
            seed = self.base_seed + i
            print(f"[Ensemble] Generating candidate {i+1}/{self.n_candidates} (seed={seed})")

            # Generate one candidate
            frames = self.pipeline.generate_variable_n(
                history_frames=history_frames,
                prompt=prompt,
                n_frames=n_frames,
                seed=seed,
                relay_inputs=relay_inputs,
                **generation_kwargs,
            )  # (n_frames, 3, H, W)

            # Fix VRAM Bottleneck: Push large candidate tensor to CPU immediately
            frames = frames.cpu()
            candidates.append(frames)

            # Score this candidate (safely offloaded)
            score_dict = self._score_candidate(
                history_frames=history_frames.cpu(),
                generated_frames=frames,
                prompt=clip_text,
            )
            scores.append(score_dict)

            print(
                f"[Ensemble] Candidate {i+1} scores: "
                f"CLIP={score_dict['clip_s']:.3f}, "
                f"Hist={score_dict['hist']:.3f}, "
                f"Temp={score_dict['temp']:.3f}, "
                f"Composite={score_dict['composite']:.3f}"
            )

        # Select best candidate
        composite_scores = [s["composite"] for s in scores]
        best_idx = int(np.argmax(composite_scores))
        best_frames = candidates[best_idx]

        print(f"[Ensemble] Best candidate: {best_idx+1} "
              f"(composite={composite_scores[best_idx]:.3f})")

        info = {
            "best_idx": best_idx,
            "all_scores": scores,
            "best_score": scores[best_idx],
            "n_candidates": self.n_candidates,
        }

        return best_frames, info

    def _score_candidate(
        self,
        history_frames: torch.Tensor,
        generated_frames: torch.Tensor,
        prompt: str,
    ) -> Dict[str, float]:
        """Score a single candidate on all metrics."""
        # CLIP-S: text-video alignment
        clip_s = 0.0
        if self.clip_scorer is not None:
            clip_s = self.clip_scorer.score(generated_frames, prompt)

        # History consistency: temporal continuity at boundary
        hist = self.hist_scorer.score(history_frames, generated_frames)

        # Temporal coherence: internal smoothness
        temp = self.temp_scorer.score(generated_frames)

        # V-JEPA: deep perceptual continuity
        vjepa = 0.5  # neutral default
        if self.vjepa_scorer is not None:
            try:
                vjepa = self.vjepa_scorer.score(history_frames, generated_frames)
            except Exception as e:
                print(f"[Ensemble] V-JEPA scoring failed: {e}")

        # Composite weighted score
        w = self.weights
        composite = (
            w["clip_s"] * clip_s
            + w["hist"] * hist
            + w["temp"] * temp
            + w["vjepa"] * vjepa
        )

        return {
            "clip_s": clip_s,
            "hist": hist,
            "temp": temp,
            "vjepa": vjepa,
            "composite": composite,
        }
