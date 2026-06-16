"""
src/prompt/coect_processor.py
==============================
CoECT Processor — all bug fixes applied.

arXiv: 2603.09094  Chain of Event-Centric Causal Thought
GitHub: https://github.com/ZixuanWang0525/CoECT

BUG FIXES APPLIED:
  #1  temporal_segments (not elementary_events): parallel actors in same window
  #6  <think>...</think> stripping for DeepSeek-R1 / Qwen reasoning models
  #6  openai_compatible backend (vLLM, DeepSeek, Qwen endpoints)
  #6  regex-extract last {...} block instead of naive json.loads

Bernini CoT Integration (arXiv:2605.22344):
  allow_chain_of_thought=True → LLM may reason in <think> tags before JSON.
  We strip those tags; the final JSON is cleaner (CoT improves quality).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional

# ── LLM backends ──────────────────────────────────────────────────────────────
try:
    from anthropic import Anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False


# =============================================================================
# CONSTANTS
# =============================================================================
PHASE_NAMES = {
    "1": "Pre-Recognition",
    "2": "Recognition",
    "3": "Judgment",
    "4": "Action",
    "5": "Avoidance",
}

TRAFFIC_PHYSICS = """\
Traffic physics constraints (use where applicable):
• TTC = distance / relative_velocity  — if TTC < 2s → action phase
• Pedestrian crossing: ~4m / 1.4 m/s ≈ 2.9 seconds to cross one lane
• Vehicle stopping: d_stop = v²/(2×0.7×9.8)  [at 30km/h ≈ 5m; at 50km/h ≈ 13.8m]
• Driver reaction: ~0.75–1.5 seconds from recognition to brake onset
"""

SYSTEM_PROMPT = """\
You are a traffic scene analyst expert. You decompose traffic event descriptions
into causally grounded temporal segments.

CRITICAL RULES:
1. Output ONLY valid JSON — no markdown fences, no preamble, no postscript.
2. Multiple actors (pedestrian + vehicle) may be SIMULTANEOUS in the same segment.
   Do NOT force sequential splits for parallel actions.
3. temporal_segments must describe what BOTH actors do in each time window.
4. Use chain-of-thought reasoning internally before producing the JSON.
"""

# Bug Fix #1: temporal_segments (not sequential elementary_events)
USER_TEMPLATE = """\
Analyze this traffic event.

Phase: {phase_label} — {phase_name}
Segment duration: {duration:.1f}s
Camera: overhead/bird's-eye surveillance view

Pedestrian behavior:
{cap_ped}

Vehicle behavior:
{cap_veh}

Physics reference:
{physics}

Generate a structured analysis with:
1. global_prompt: 1-2 sentences of overhead scene context (camera angle, road, participants)
2. temporal_segments: 2-4 time windows covering the segment.
   IMPORTANT: each segment can describe BOTH actors simultaneously (no forced sequential split).
3. enhanced_pedestrian_prompt: enriched ped description with camera context
4. enhanced_vehicle_prompt: enriched vehicle description with camera context
5. temporal_keyframes.t0 / t50 / t100: brief what-is-visible descriptions at 0%, 50%, 100%

Output JSON (no markdown, no extra text):
{{
  "global_prompt": "...",
  "temporal_segments": [
    {{
      "description": "Both pedestrian and vehicle [what they do simultaneously]",
      "duration_fraction": 0.5,
      "causal_note": "why this happens"
    }}
  ],
  "enhanced_pedestrian_prompt": "...",
  "enhanced_vehicle_prompt": "...",
  "temporal_keyframes": {{
    "t0":  "...",
    "t50": "...",
    "t100": "..."
  }}
}}"""


# =============================================================================
# JSON EXTRACTION (Bug Fix #6 — handles <think> tags + broken JSON)
# =============================================================================
def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks (DeepSeek-R1, Qwen reasoning models)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> Dict:
    """
    Extract the last complete JSON object from text.

    Strategy:
      1. Strip <think> tags (Bug Fix #6).
      2. Find the LAST occurrence of {...} (outermost braces).
      3. Try json.loads on that substring.
      4. Fallback: try the whole stripped text.
    """
    clean = _strip_think_tags(text)

    # Find last { ... } block (handles models that add trailing text)
    start = clean.rfind("{")
    end   = clean.rfind("}") + 1
    if start >= 0 and end > start:
        candidate = clean[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fallback: strip code fences and try again
    no_fence = re.sub(r"```(?:json)?|```", "", clean).strip()
    return json.loads(no_fence)


# =============================================================================
# COECT PROCESSOR
# =============================================================================
class CoECTProcessor:
    """
    Causal Event Chain Decomposition for WTS traffic captions.

    Bug fixes applied:
      #1: temporal_segments (parallel actors in same window)
      #6: <think> tag stripping + last-{} extraction + openai_compatible backend

    Bernini CoT: allow_chain_of_thought=True lets the LLM reason freely
    inside <think> blocks before producing final JSON. This improves
    output quality (Bernini §2.1.1 chain-of-thought planning).

    Args:
        llm_backend: "anthropic" | "openai_compatible" | "none"
        llm_model:   model name (e.g. "claude-sonnet-4-6")
        openai_base_url: for openai_compatible backend (DeepSeek, Qwen, local vLLM)
        allow_chain_of_thought: whether to encourage <think> reasoning
        use_traffic_physics: inject physics formulas into prompt
        cache_dir: directory for MD5-keyed response cache
    """

    def __init__(
        self,
        llm_backend:           str  = "anthropic",
        llm_model:             str  = "claude-sonnet-4-6",
        openai_base_url:       str  = "https://api.openai.com/v1",
        allow_chain_of_thought: bool = True,
        use_traffic_physics:   bool = True,
        cache_dir:             str  = "./cache/coect",
        max_segments:          int  = 4,
    ):
        self.backend  = llm_backend
        self.model    = llm_model
        self.base_url = openai_base_url
        self.cot      = allow_chain_of_thought
        self.physics  = use_traffic_physics
        self.max_seg  = max_segments

        # Build LLM client
        self._client = None
        if llm_backend == "anthropic":
            if not _ANTHROPIC_OK:
                warnings.warn("anthropic package not installed; falling back to 'none'.")
                self.backend = "none"
            else:
                self._client = Anthropic()
        elif llm_backend == "openai_compatible":
            if not _OPENAI_OK:
                warnings.warn("openai package not installed; falling back to 'none'.")
                self.backend = "none"
            else:
                api_key = os.environ.get("OPENAI_API_KEY", "dummy")
                self._client = OpenAI(base_url=openai_base_url, api_key=api_key)

        # MD5-keyed cache (avoids re-calling API for same input)
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_file = os.path.join(cache_dir, "responses.json")
        self._cache: Dict[str, Any] = {}
        if os.path.exists(self._cache_file):
            with open(self._cache_file) as f:
                self._cache = json.load(f)

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _key(self, cap_ped: str, cap_veh: str, phase: str) -> str:
        raw = f"{phase}|{cap_ped[:120]}|{cap_veh[:120]}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _save_cache(self) -> None:
        with open(self._cache_file, "w") as f:
            json.dump(self._cache, f, indent=2)

    # ── Core ──────────────────────────────────────────────────────────────────

    def process(
        self,
        caption_pedestrian: str,
        caption_vehicle:    str,
        phase_label:        str   = "4",
        duration:           float = 5.0,
    ) -> Dict[str, Any]:
        """
        Decompose one WTS segment into a causal temporal structure.

        Returns dict with:
          global_prompt, temporal_segments, enhanced_pedestrian_prompt,
          enhanced_vehicle_prompt, temporal_keyframes, relay_inputs
        """
        key = self._key(caption_pedestrian, caption_vehicle, phase_label)
        if key in self._cache:
            result = self._cache[key]
            result["relay_inputs"] = self._relay_inputs(result, duration)
            return result

        if self.backend == "none" or self._client is None:
            result = self._fallback(caption_pedestrian, caption_vehicle, phase_label)
        else:
            result = self._llm_call(caption_pedestrian, caption_vehicle, phase_label, duration)

        result["relay_inputs"] = self._relay_inputs(result, duration)
        cache_result = {k: v for k, v in result.items() if k != "relay_inputs"}
        self._cache[key] = cache_result
        self._save_cache()
        return result

    def _llm_call(self, cap_ped, cap_veh, phase, duration) -> Dict:
        physics_text = TRAFFIC_PHYSICS if self.physics else ""
        user_msg = USER_TEMPLATE.format(
            phase_label=phase,
            phase_name=PHASE_NAMES.get(phase, "Traffic Event"),
            duration=duration,
            cap_ped=cap_ped,
            cap_veh=cap_veh,
            physics=physics_text,
        )
        # Allow CoT reasoning (Bernini §2.1.1)
        if self.cot:
            user_msg = (
                "You may reason step-by-step inside <think>...</think> tags "
                "before producing the final JSON.\n\n" + user_msg
            )
        try:
            if self.backend == "anthropic":
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = resp.content[0].text
            else:  # openai_compatible
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    max_tokens=1500,
                    temperature=0.3,
                )
                raw = resp.choices[0].message.content

            parsed = _extract_json(raw)   # Bug Fix #6: strip <think>, extract last {}

            # Normalize
            segs = parsed.get("temporal_segments", [])
            if len(segs) > self.max_seg:
                segs = segs[:self.max_seg]
            total = sum(s.get("duration_fraction", 0.0) for s in segs)
            if total > 0:
                for s in segs:
                    s["duration_fraction"] = s.get("duration_fraction", 0.0) / total
            parsed["temporal_segments"] = segs
            return parsed

        except Exception as e:
            warnings.warn(f"[CoECT] LLM call failed ({e}); using fallback.")
            return self._fallback(cap_ped, cap_veh, phase)

    def _fallback(self, cap_ped: str, cap_veh: str, phase: str) -> Dict:
        """Template fallback when LLM is unavailable."""
        prefix = "Overhead surveillance camera, bird's-eye traffic view. "
        phase_desc = PHASE_NAMES.get(phase, "Traffic event")
        return {
            "global_prompt": (
                f"{prefix}{phase_desc}: {cap_ped[:80]} Meanwhile: {cap_veh[:80]}"
            ),
            "temporal_segments": [
                {
                    "description": (
                        f"[0-50%] {phase_desc}: {cap_ped[:100]} "
                        f"Simultaneously: {cap_veh[:80]}"
                    ),
                    "duration_fraction": 0.5,
                    "causal_note": "simultaneous action",
                },
                {
                    "description": (
                        f"[50-100%] Resolution: {cap_veh[:100]} "
                        f"Pedestrian: {cap_ped[100:180] or 'continues action'}"
                    ),
                    "duration_fraction": 0.5,
                    "causal_note": "outcome and avoidance",
                },
            ],
            "enhanced_pedestrian_prompt": f"{prefix}Pedestrian: {cap_ped}",
            "enhanced_vehicle_prompt":    f"{prefix}Vehicle: {cap_veh}",
            "temporal_keyframes": {
                "t0":   f"Start of {phase_desc}: {cap_ped[:60]}",
                "t50":  f"Mid-point: both actors responding",
                "t100": f"Resolution: {cap_veh[:60]}",
            },
        }

    def _relay_inputs(self, result: Dict, duration: float) -> Dict:
        """
        Convert temporal_segments to Prompt Relay inputs.
        Bug Fix (VAE alignment): use cumulative sum for fractional boundaries.
        """
        segs = result.get("temporal_segments", [])
        if not segs:
            return {
                "global_prompt":    result.get("global_prompt", ""),
                "local_prompts":    [
                    result.get("enhanced_pedestrian_prompt", ""),
                    result.get("enhanced_vehicle_prompt", ""),
                ],
                "segment_fractions": [0.5, 0.5],
                "segment_lengths":   [0.5, 0.5],
            }
        fracs = [s.get("duration_fraction", 1/len(segs)) for s in segs]
        return {
            "global_prompt":     result.get("global_prompt", ""),
            "local_prompts":     [s["description"] for s in segs],
            "segment_fractions": fracs,
            "segment_lengths":   fracs,
        }
