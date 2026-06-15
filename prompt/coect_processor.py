"""
src/prompt/coect_processor.py
==============================
CoECT – Chain of Event-Centric Causal Thought Processor

This module implements causal event decomposition for traffic scenarios,
inspired by the CoECT paper (arXiv: 2603.09094).

CoECT Reference
---------------
Paper: "Chain of Event-Centric Causal Thought for Physically Plausible
        Video Generation" – Wang et al. (Sichuan University, 2025)
Code:  https://github.com/ZixuanWang0525/CoECT
arXiv: https://arxiv.org/pdf/2603.09094

Key idea from CoECT:
  1. CECD (Causal Event Chain Decomposition): decompose complex physical
     phenomena into elementary event units using chain-of-thought + physics
     formula constraints for causal determinism.
  2. TCP (Transition-aware Cross-modal Prompting): generate causally
     coherent vision-language prompts + intermediate keyframe descriptions.

Adaptation for WTS Traffic Domain
----------------------------------
Original CoECT targets physical phenomena (pendulum, projectile, etc.).
We adapt it to traffic events:
  - Elementary events: vehicle decelerating, pedestrian stepping off curb, etc.
  - Physics constraints: collision avoidance, velocity × time = distance, etc.
  - Traffic-specific formulas: TTC (Time To Collision) = distance / relative_velocity

Why CoECT for Traffic?
-----------------------
WTS annotations describe 5 event phases:
  Pre-recognition → Recognition → Judgment → Action → Avoidance

Each caption describes what happens in that phase, but NOT the causal chain.
CoECT helps the LLM reason:
  "The vehicle brakes BECAUSE it recognized the pedestrian at t=2s; therefore
   by t=4s the pedestrian should be visible in frame center..."

This causally grounded prompt → better temporal coherence → higher PSNR/SSIM.

Integration with Prompt Relay
------------------------------
CoECT outputs:
  - global_prompt: overall scene context (fed as Prompt Relay global)
  - event_segments: list of (local_prompt, duration) pairs (fed as Prompt Relay locals)
  - keyframe_descriptions: intermediate frame descriptions (optional conditioning)
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# WTS-specific traffic physics formulas for causal constraints
# (adapted from CoECT's formula-constrained causal reasoning)
# ---------------------------------------------------------------------------

TRAFFIC_PHYSICS_FORMULAS = """
Key physics constraints for traffic scenarios:
1. Time-To-Collision (TTC): TTC = distance / relative_velocity
   - If TTC < 2s → vehicle is braking/swerving (Action phase)
   - If TTC = 2-5s → vehicle is recognizing/judging (Recognition, Judgment phase)
2. Stopping distance: d_stop = v² / (2μg), μ ≈ 0.7 (dry road)
   - At 30 km/h (8.3 m/s): d_stop ≈ 5.0 m
   - At 50 km/h (13.9 m/s): d_stop ≈ 13.8 m
3. Pedestrian crossing time: t_cross = lane_width / pedestrian_speed
   - Typical: 4m lane / 1.4 m/s walking = ~2.9 seconds to cross
4. Vehicle deceleration: a_brake ≈ -5 to -8 m/s² (panic braking)
5. Reaction time: t_react ≈ 0.75-1.5s from recognition to braking
"""

# Phase-to-description mapping (for enriching raw captions)
PHASE_DESCRIPTIONS = {
    "1": "Pre-recognition phase: pedestrian and vehicle approaching, no awareness yet established",
    "2": "Recognition phase: driver begins to notice the pedestrian's presence",
    "3": "Judgment phase: driver assessing risk and deciding on response action",
    "4": "Action phase: driver actively braking, swerving, or pedestrian stepping back",
    "5": "Avoidance phase: near-miss or safe passage, collision successfully avoided",
}


# ---------------------------------------------------------------------------
# Prompt templates for CoECT-style causal reasoning
# ---------------------------------------------------------------------------

COECT_SYSTEM_PROMPT = """You are an expert traffic scene analyst. You decompose traffic
event descriptions into causally ordered elementary events with physics constraints.

You understand the 5-phase WTS annotation framework:
  Phase 1 (Pre-Recognition): No awareness between participants
  Phase 2 (Recognition): Driver first sees/notices pedestrian
  Phase 3 (Judgment): Driver assesses and decides response
  Phase 4 (Action): Physical response (braking, swerving, stepping back)
  Phase 5 (Avoidance): Resolution (near-miss, safe crossing, stop)

CRITICAL: Your final output must be valid JSON. No markdown backticks, no extra text.
If your architecture supports reasoning, you may use <think>...</think> tags before the JSON.
"""

COECT_USER_TEMPLATE = """Analyze this traffic event segment (Phase {phase_label}: {phase_name}).

Camera: Overhead/bird's-eye view of a traffic intersection or road.
Segment duration: {duration:.1f} seconds

Pedestrian behavior:
{caption_pedestrian}

Vehicle behavior:
{caption_vehicle}

Motion Context (from Optical Flow):
{motion_context}

Relevant physics:
{physics_formulas}

Decompose into temporal segments and generate:
1. global_prompt: Single paragraph describing the overall scene context
   (camera angle, setting, participants) for the whole video.
2. temporal_segments: List of 1-3 temporal segments. IMPORTANT: You MUST group events that happen at the same time into the SAME segment.
   Each segment has:
   - start_fraction: start time (0.0 to 1.0)
   - end_fraction: end time (0.0 to 1.0, must connect to next start_fraction)
   - scene_description: what happens (FUSE simultaneous vehicle and pedestrian actions together using "WHILE" or "MEANWHILE")
   - active_physics: relevant formula/constraint (or "none")
3. enhanced_pedestrian_prompt: Enriched prompt for pedestrian-focused frames
4. enhanced_vehicle_prompt: Enriched prompt for vehicle-focused frames
5. temporal_keyframes: 3 descriptions of what frames should look like at
   t=0% (start), t=50% (middle), t=100% (end) of the segment

Output format (JSON):
{{
  "global_prompt": "...",
  "temporal_segments": [
    {{
      "start_fraction": 0.0,
      "end_fraction": 0.5,
      "scene_description": "A vehicle applies hard brakes (due to TTC < 2s) WHILE a pedestrian simultaneously steps back onto the curb.",
      "active_physics": "d_stop = v^2/(2*mu*g)"
    }},
    {{
      "start_fraction": 0.5,
      "end_fraction": 1.0,
      "scene_description": "The vehicle comes to a complete halt just before the crosswalk, and the pedestrian remains stationary.",
      "active_physics": "v = 0"
    }}
  ],
  "enhanced_pedestrian_prompt": "...",
  "enhanced_vehicle_prompt": "...",
  "temporal_keyframes": {{
    "t0_description": "...",
    "t50_description": "...",
    "t100_description": "..."
  }}
}}"""


# ---------------------------------------------------------------------------
# Main CoECT Processor Class
# ---------------------------------------------------------------------------

class CoECTProcessor:
    """
    Causal Event Chain Decomposition for traffic video generation.
    
    Implements the CoECT (arXiv 2603.09094) framework adapted for the
    WTS traffic domain. Uses an LLM to perform chain-of-thought causal
    reasoning over traffic event phase descriptions.
    
    The output feeds directly into PromptRelayRouter:
      - global_prompt → Prompt Relay global (scene context throughout)
      - elementary_events → Prompt Relay local prompts (each gets a temporal window)
      - enhanced_{ped,veh}_prompt → used for CLIP-S scoring
    
    Args:
        llm_backend: "anthropic" | "openai_compatible" | "none" (skip LLM, use template)
        llm_model: model name (e.g. "claude-sonnet-4-6" or "deepseek-reasoner")
        use_traffic_physics: whether to include physics formulas in the prompt
        cache_dir: path to cache LLM responses (avoid re-calling API)
        
    Example:
        >>> processor = CoECTProcessor(llm_backend="anthropic")
        >>> result = processor.process(
        ...     caption_pedestrian="The pedestrian stands still...",
        ...     caption_vehicle="The vehicle was positioned diagonally...",
        ...     phase_label="4",
        ...     duration=5.268,
        ... )
        >>> print(result["global_prompt"])
        >>> print(result["elementary_events"])  # list of dicts
    """
    
    def __init__(
        self,
        llm_backend: str = "anthropic",
        llm_model: str = "claude-sonnet-4-6",
        use_traffic_physics: bool = True,
        cache_dir: Optional[str] = "./cache/coect",
        max_events: int = 5,
    ):
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.use_traffic_physics = use_traffic_physics
        self.max_events = max_events
        
        # LLM client
        self._client = None
        if llm_backend == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic()
        elif llm_backend == "openai_compatible":
            from openai import OpenAI
            # Expects OPENAI_API_KEY and OPENAI_BASE_URL in environment for vLLM/Ollama
            self._client = OpenAI()
        
        # Response cache (saves API calls between runs)
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        self._response_cache: Dict[str, Any] = {}
        self._load_cache()
    
    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    
    def _cache_key(self, caption_ped: str, caption_veh: str, phase: str, motion_context: Optional[Dict[str, Any]] = None) -> str:
        """Deterministic cache key from inputs."""
        import hashlib
        motion_str = json.dumps(motion_context, sort_keys=True) if motion_context else ""
        combined = f"{phase}|{caption_ped}|{caption_veh}|{motion_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _load_cache(self) -> None:
        """Load cached LLM responses from disk."""
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, "responses.json")
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    self._response_cache = json.load(f)
    
    def _save_cache(self) -> None:
        """Persist response cache to disk."""
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, "responses.json")
            with open(cache_file, "w") as f:
                json.dump(self._response_cache, f, indent=2)
    
    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    
    def process(
        self,
        caption_pedestrian: str,
        caption_vehicle: str,
        phase_label: str = "4",
        duration: float = 5.0,
        motion_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point: decompose traffic event into causal chain.
        
        Args:
            caption_pedestrian: WTS pedestrian behavior description
            caption_vehicle: WTS vehicle behavior description
            phase_label: WTS event phase ("1"-"5")
            duration: segment duration in seconds (end_time - start_time)
        
        Returns:
            dict with keys:
              - global_prompt: str (full scene context, used as Prompt Relay global)
              - elementary_events: List[Dict] (causal chain events)
              - enhanced_pedestrian_prompt: str
              - enhanced_vehicle_prompt: str
              - temporal_keyframes: Dict with t0/t50/t100 descriptions
              - prompt_relay_inputs: Dict ready for PromptRelayRouter
        """
        # Check cache first
        cache_key = self._cache_key(caption_pedestrian, caption_vehicle, phase_label, motion_context)
        if cache_key in self._response_cache:
            result = self._response_cache[cache_key]
            # Rebuild prompt_relay_inputs (always fresh, depends on duration)
            result["prompt_relay_inputs"] = self._build_relay_inputs(
                result, duration
            )
            return result
        
        # Call LLM for causal decomposition
        if self.llm_backend == "none" or self._client is None:
            result = self._fallback_decomposition(
                caption_pedestrian, caption_vehicle, phase_label, duration
            )
        else:
            result = self._llm_decomposition(
                caption_pedestrian, caption_vehicle, phase_label, duration, motion_context
            )
        
        # Build Prompt Relay inputs from decomposed events
        result["prompt_relay_inputs"] = self._build_relay_inputs(result, duration)
        
        # Cache (without prompt_relay_inputs since duration varies)
        cache_result = {k: v for k, v in result.items() if k != "prompt_relay_inputs"}
        self._response_cache[cache_key] = cache_result
        self._save_cache()
        
        return result
    
    def _llm_decomposition(
        self,
        caption_pedestrian: str,
        caption_vehicle: str,
        phase_label: str,
        duration: float,
        motion_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call LLM (Claude) for causal event chain decomposition.
        Implements CoECT's CECD + TCP pipeline.
        """
        phase_name = PHASE_DESCRIPTIONS.get(phase_label, "Unknown phase")
        physics = TRAFFIC_PHYSICS_FORMULAS if self.use_traffic_physics else ""
        motion_str = json.dumps(motion_context, indent=2) if motion_context else "None"
        
        user_msg = COECT_USER_TEMPLATE.format(
            phase_label=phase_label,
            phase_name=phase_name,
            duration=duration,
            caption_pedestrian=caption_pedestrian,
            caption_vehicle=caption_vehicle,
            motion_context=motion_str,
            physics_formulas=physics,
        )
        
        try:
            if self.llm_backend == "anthropic":
                response = self._client.messages.create(
                    model=self.llm_model,
                    max_tokens=1500,
                    system=COECT_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw_text = response.content[0].text.strip()
            elif self.llm_backend == "openai_compatible":
                response = self._client.chat.completions.create(
                    model=self.llm_model,
                    max_tokens=1500,
                    messages=[
                        {"role": "system", "content": COECT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ],
                )
                raw_text = response.choices[0].message.content.strip()
            
            # Robust JSON Parsing for Reasoning Models (DeepSeek-R1 / Qwen2.5)
            # 1. Strip out <think>...</think> blocks entirely
            raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
            
            # 2. Extract JSON (from first { to last })
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                raw_text = match.group(0)
            
            # Parse JSON response
            result = json.loads(raw_text)
            
            # Validate and clip to max_events
            if len(result.get("temporal_segments", [])) > self.max_events:
                result["temporal_segments"] = result["temporal_segments"][:self.max_events]
            
            # Normalize fractions to exactly 1.0 span
            segments = result.get("temporal_segments", [])
            for i, seg in enumerate(segments):
                # Ensure sequential connectivity
                if i == 0:
                    seg["start_fraction"] = 0.0
                else:
                    seg["start_fraction"] = segments[i-1].get("end_fraction", 0.0)
                
                # Cap the final segment at 1.0
                if i == len(segments) - 1:
                    seg["end_fraction"] = 1.0
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"[CoECT] JSON parse error: {e}. Falling back to template.")
            return self._fallback_decomposition(
                caption_pedestrian, caption_vehicle, phase_label, duration
            )
        except Exception as e:
            print(f"[CoECT] LLM call failed: {e}. Falling back to template.")
            return self._fallback_decomposition(
                caption_pedestrian, caption_vehicle, phase_label, duration
            )
    
    def _fallback_decomposition(
        self,
        caption_pedestrian: str,
        caption_vehicle: str,
        phase_label: str,
        duration: float,
    ) -> Dict[str, Any]:
        """
        Template-based fallback when LLM is unavailable.
        
        Creates a simple 2-event decomposition:
          Event 1 (0-50%): pedestrian-focused description
          Event 2 (50-100%): vehicle-focused description
        """
        phase_name = PHASE_DESCRIPTIONS.get(phase_label, "traffic event")
        
        # Build global prompt with camera context
        global_prompt = (
            f"Overhead bird's-eye view of a traffic scenario captured by a surveillance "
            f"camera. {phase_name}. "
            f"Scene shows: {caption_pedestrian[:150]} "
            f"Meanwhile: {caption_vehicle[:150]}"
        )
        
        # Enhanced prompts (add camera viewpoint context)
        ped_prompt = (
            f"Overhead traffic camera view. Pedestrian behavior: {caption_pedestrian} "
            f"Traffic phase: {phase_name}."
        )
        veh_prompt = (
            f"Overhead traffic camera view. Vehicle behavior: {caption_vehicle} "
            f"Traffic phase: {phase_name}."
        )
        
        # Single 100% temporal segment Fallback
        temporal_segments = [
            {
                "start_fraction": 0.0,
                "end_fraction": 1.0,
                "scene_description": f"Traffic event unfolding simultaneously: {caption_pedestrian} Meanwhile, {caption_vehicle}",
                "active_physics": "none",
            }
        ]
        
        return {
            "global_prompt": global_prompt,
            "temporal_segments": temporal_segments,
            "enhanced_pedestrian_prompt": ped_prompt,
            "enhanced_vehicle_prompt": veh_prompt,
            "temporal_keyframes": {
                "t0_description": f"Start of {phase_name}: {caption_pedestrian[:80]}",
                "t50_description": f"Mid-point: transition between pedestrian and vehicle action",
                "t100_description": f"End of {phase_name}: {caption_vehicle[:80]}",
            },
        }
    
    def _build_relay_inputs(
        self,
        coect_result: Dict[str, Any],
        duration: float,
    ) -> Dict[str, Any]:
        """
        Convert CoECT output into Prompt Relay input format.
        
        Prompt Relay expects:
          - global_prompt: anchors persistent scene elements
          - local_prompts: list of event-specific descriptions
          - segment_lengths: number of frames for each local prompt
          
        This mapping enables each CoECT elementary event to guide only
        its designated temporal window → reduces semantic interference.
        
        See: arXiv 2604.10030 (Prompt Relay), Section 3
        """
        segments = coect_result.get("temporal_segments", [])
        
        if not segments:
            # Single-event fallback
            return {
                "global_prompt": coect_result.get("global_prompt", ""),
                "local_prompts": [
                    f"{coect_result.get('enhanced_pedestrian_prompt', '')} Meanwhile, {coect_result.get('enhanced_vehicle_prompt', '')}"
                ],
                "segment_lengths": [1.0],
            }
        
        local_prompts = [seg.get("scene_description", "") for seg in segments]
        segment_lengths = [max(0.0, seg.get("end_fraction", 1.0) - seg.get("start_fraction", 0.0)) for seg in segments]
        
        return {
            "global_prompt": coect_result.get("global_prompt", ""),
            "local_prompts": local_prompts,
            "segment_lengths": segment_lengths,
        }
    
    def batch_process(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of WTS samples.
        
        Args:
            samples: list of dicts, each with:
                caption_pedestrian, caption_vehicle, phase_label, duration
        
        Returns:
            List of CoECT result dicts
        """
        results = []
        for i, sample in enumerate(samples):
            print(f"[CoECT] Processing sample {i+1}/{len(samples)}...")
            result = self.process(
                caption_pedestrian=sample["caption_pedestrian"],
                caption_vehicle=sample["caption_vehicle"],
                phase_label=str(sample.get("phase_label", "4")),
                duration=float(sample.get("duration", 5.0)),
            )
            results.append(result)
        return results
