"""
src/prompt/prompt_builder.py
==============================
Unified Prompt Construction Pipeline

Orchestrates all prompt-related components into a single interface:
  1. CoECT causal event decomposition (arXiv 2603.09094)
  2. Prompt Relay temporal routing (arXiv 2604.10030)
  3. Motion-context enrichment (our adaptation of WAN Animate insight)
  4. Traffic-domain template wrapping

This module is the single entry point for all prompt processing.
The run_inference.py script calls PromptBuilder.build() for each test case.

Output Structure (for the generation pipeline):
    {
      "combined_prompt": str,              # full text for TI2V input
      "global_prompt": str,               # scene-wide context
      "local_prompts": List[str],         # per-segment event descriptions
      "segment_lengths": List[float],     # fractional lengths per segment
      "enhanced_ped_prompt": str,         # enriched pedestrian prompt
      "enhanced_veh_prompt": str,         # enriched vehicle prompt
      "motion_text": str,                 # motion context from optical flow
      "relay_inputs": Dict,               # ready-to-use Prompt Relay args
    }

References:
    CoECT:        arXiv 2603.09094 – causal event chain decomposition
    Prompt Relay: arXiv 2604.10030 – inference-time temporal routing
    WAN Animate:  https://humanaigc.github.io/wan-animate/ (motion insight)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from prompt.coect_processor import CoECTProcessor
from utils.motion_analyzer import MotionAnalyzer, MotionFeatures


# ═══════════════════════════════════════════════════════════════════
# TRAFFIC-SPECIFIC PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════

OVERHEAD_CAMERA_PREFIX = (
    "Bird's-eye overhead surveillance camera view of a traffic scenario. "
    "The camera is fixed, looking straight down at a road intersection "
    "or crosswalk. "
)

TRAFFIC_SCENE_QUALITY = (
    "High-definition, sharp, clear footage, realistic traffic scene, "
    "accurate perspective, consistent lighting."
)

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, "
    "static overall gray, worst quality, low quality, JPEG artifacts, "
    "watermark, logo, distorted perspective, aerial 3D view, satellite, "
    "cartoon, anime, painting, artwork."
)

# WTS event phase descriptions (prepended to prompt for phase-awareness)
PHASE_PREFIXES = {
    "1": "Pre-recognition traffic scenario: ",
    "2": "Traffic recognition moment: ",
    "3": "Traffic judgment phase: ",
    "4": "Traffic action phase — response in progress: ",
    "5": "Traffic avoidance and resolution: ",
}


# ═══════════════════════════════════════════════════════════════════
# COMBINE STRATEGIES
# ═══════════════════════════════════════════════════════════════════

def combine_captions_optimal(
    cap_ped: str, cap_veh: str, phase_label: str = "4"
) -> str:
    """
    Optimal Token Positioning: Action first, Style last.
    Converts text to natural prose, removing rigid ':' separators.
    """
    phase_context = PHASE_PREFIXES.get(phase_label, "").lower()
    
    # 1. Core Action (Highest Attention Weight)
    # Using "while" to signal simultaneous/parallel events
    core_action = f"A {phase_context} where {cap_veh.lower()} while {cap_ped.lower()}"
    
    # 2. Camera and Style (Secondary Weight)
    camera_style = (
        "Viewed from a fixed, bird's-eye overhead surveillance camera at an intersection. "
        "High-definition, realistic traffic scene."
    )
    
    return f"{core_action}. {camera_style}"


def combine_captions_relay(
    cap_ped: str,
    cap_veh: str,
    global_context: str,
    phase_label: str = "4",
) -> Dict[str, Any]:
    """
    Prompt Relay strategy: route ped caption to first half, veh to second half.

    This is the MINIMUM viable Prompt Relay setup when CoECT is unavailable.
    CoECT provides a richer decomposition; this is the fallback.

    Returns dict with keys: global_prompt, local_prompts, segment_lengths
    """
    return {
        "global_prompt": global_context or f"{OVERHEAD_CAMERA_PREFIX}{PHASE_PREFIXES.get(phase_label, '')}",
        "local_prompts": [cap_ped, cap_veh],
        "segment_lengths": [0.45, 0.55],  # ped first, veh second (slight veh bias)
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════

class PromptBuilder:
    """
    Unified prompt construction for WTS Track 5.

    Combines:
      - CoECT: LLM-based causal event decomposition
      - Prompt Relay inputs: global + local prompts + segment lengths
      - Motion enrichment: optical flow context appended to prompt
      - Camera/domain context: overhead view, traffic scene

    Three modes depending on available resources:
      Mode 1 (BEST):    CoECT + Prompt Relay + Motion Enrichment
      Mode 2 (FAST):    Simple combine + Prompt Relay + Motion Enrichment
      Mode 3 (MINIMAL): Simple concatenation only

    Args:
        use_coect: whether to use LLM-based CoECT decomposition
        use_motion_enrichment: whether to append optical flow context
        use_prompt_relay: whether to output relay-ready inputs
        coect_llm_backend: "anthropic" | "openai" | "none"
        coect_llm_model: LLM model name
        flow_method: "farneback" (CPU) | "raft" (GPU)
        flow_device: device for RAFT flow estimation
        coect_cache_dir: directory to cache LLM responses
    """

    def __init__(
        self,
        use_coect: bool = True,
        use_motion_enrichment: bool = True,
        use_prompt_relay: bool = True,
        coect_llm_backend: str = "anthropic",
        coect_llm_model: str = "claude-sonnet-4-6",
        flow_method: str = "farneback",
        flow_device: str = "cpu",
        coect_cache_dir: str = "./cache/coect",
    ):
        self.use_coect = use_coect
        self.use_motion_enrichment = use_motion_enrichment
        self.use_prompt_relay = use_prompt_relay

        # Initialize CoECT processor
        if use_coect:
            self.coect = CoECTProcessor(
                llm_backend=coect_llm_backend,
                llm_model=coect_llm_model,
                use_traffic_physics=True,
                cache_dir=coect_cache_dir,
            )
        else:
            self.coect = None

        # Initialize motion analyzer
        if use_motion_enrichment:
            self.motion_analyzer = MotionAnalyzer(
                flow_method=flow_method,
                device=flow_device,
            )
        else:
            self.motion_analyzer = None

    def build(
        self,
        caption_pedestrian: str,
        caption_vehicle: str,
        phase_label: str = "4",
        duration: float = 5.0,
        history_frames=None,        # optional: torch.Tensor (K, 3, H, W)
        n_output_frames: int = 33,  # how many frames to generate (for relay split)
    ) -> Dict[str, Any]:
        """
        Build complete prompt package for one WTS test case.

        Args:
            caption_pedestrian: raw WTS pedestrian caption
            caption_vehicle: raw WTS vehicle caption
            phase_label: WTS event phase ("1"-"5")
            duration: segment duration in seconds
            history_frames: optional tensor (K, 3, H, W) for motion analysis
            n_output_frames: number of output frames (used for relay timing)

        Returns:
            Dict with all prompt components:
                combined_prompt:    str  → use this as pipe(prompt=...)
                global_prompt:      str
                local_prompts:      List[str]
                segment_lengths:    List[float]  (sum ≈ 1.0)
                enhanced_ped_prompt: str
                enhanced_veh_prompt: str
                motion_text:        str  (empty if no motion analysis)
                coect_result:       Dict  (full CoECT output, or None)
                negative_prompt:    str
        """
        # ── Step 1: Analyze motion from history frames ───────────────
        motion_features: Optional[MotionFeatures] = None
        motion_dict = {}

        if self.motion_analyzer is not None and history_frames is not None:
            try:
                motion_features = self.motion_analyzer.analyze(history_frames)
                motion_dict = motion_features.to_dict()
            except Exception as e:
                print(f"[PromptBuilder] Motion analysis failed: {e}")

        # ── Step 2: CoECT causal decomposition ──────────────────────
        coect_result = None
        if self.coect is not None:
            try:
                coect_result = self.coect.process(
                    caption_pedestrian=caption_pedestrian,
                    caption_vehicle=caption_vehicle,
                    phase_label=phase_label,
                    duration=duration,
                    motion_context=motion_dict, # Pass structured motion features directly to CoECT LLM
                )
            except Exception as e:
                print(f"[PromptBuilder] CoECT failed: {e}")

        # ── Step 3: Build relay inputs ───────────────────────────────
        if coect_result is not None:
            # Use CoECT's decomposition for Prompt Relay
            relay_inputs = coect_result.get("prompt_relay_inputs", {})
            global_prompt = coect_result.get("global_prompt", "")
            local_prompts = relay_inputs.get("local_prompts", [caption_pedestrian, caption_vehicle])
            segment_lengths = relay_inputs.get("segment_lengths", [0.45, 0.55])
            enhanced_ped = coect_result.get("enhanced_pedestrian_prompt", caption_pedestrian)
            enhanced_veh = coect_result.get("enhanced_vehicle_prompt", caption_vehicle)
        else:
            # Fallback: Optimal single-prompt setup
            # CRITICAL FIX: No Temporal Prompt Relay for parallel Ped/Veh events!
            optimal_combined = combine_captions_optimal(caption_pedestrian, caption_vehicle, phase_label)
            global_prompt = optimal_combined
            local_prompts = [optimal_combined]
            segment_lengths = [1.0]
            enhanced_ped = caption_pedestrian
            enhanced_veh = caption_vehicle

        # ── Step 4: Add camera + domain context ──────────────────────
        camera_context = OVERHEAD_CAMERA_PREFIX
        phase_prefix = PHASE_PREFIXES.get(phase_label, "")

        # Enrich global prompt with camera context if not already present
        if "overhead" not in global_prompt.lower() and "bird" not in global_prompt.lower():
            global_prompt = camera_context + phase_prefix + global_prompt

        # ── Step 5: (Removed) Semantic Collision ──────────────────────
        # We no longer lazily concatenate "objects are moving..." to avoid
        # global hallucination. CoECT already fused the motion_dict into the
        # specific local prompts during Step 2.

        # ── Step 6: Build combined prompt (for non-relay generation) ─
        # This is what gets fed to TI2V when Prompt Relay is not available
        # or as fallback
        if coect_result is None:
            combined_prompt = global_prompt # which is optimal_combined
        else:
            sep = ". "
            combined_parts = [global_prompt]
            combined_parts.extend(local_prompts)
            combined_parts.append(TRAFFIC_SCENE_QUALITY)
            combined_prompt = sep.join(p.strip() for p in combined_parts if p.strip())

        # ── Step 7: Truncate to reasonable length ────────────────────
        # T5 tokenizer max is 512 tokens; ~2.5 chars/token → ~1280 chars
        # Be conservative and cap at 1000 chars for safety
        if len(combined_prompt) > 1000:
            combined_prompt = combined_prompt[:997] + "..."

        return {
            "combined_prompt": combined_prompt,
            "global_prompt": global_prompt,
            "local_prompts": local_prompts,
            "segment_lengths": segment_lengths,
            "enhanced_ped_prompt": enhanced_ped,
            "enhanced_veh_prompt": enhanced_veh,
            "motion_features": motion_features,
            "coect_result": coect_result,
            "negative_prompt": NEGATIVE_PROMPT,
            # Relay-ready dict for PromptRelayRouter
            "relay_inputs": {
                "global_prompt": global_prompt,
                "local_prompts": local_prompts,
                "segment_lengths": segment_lengths,
                "num_frames": n_output_frames,
            },
        }

    def build_batch(
        self,
        samples: list,   # list of dicts with caption_pedestrian, caption_vehicle, etc.
    ) -> list:
        """Process a batch of samples."""
        return [
            self.build(
                caption_pedestrian=s["caption_pedestrian"],
                caption_vehicle=s["caption_vehicle"],
                phase_label=str(s.get("phase_label", "4")),
                duration=float(s.get("duration", 5.0)),
                history_frames=s.get("history_frames", None),
                n_output_frames=s.get("num_target_frames", 33),
            )
            for s in samples
        ]
