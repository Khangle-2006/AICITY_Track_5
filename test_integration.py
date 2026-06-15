"""
test_integration.py
====================
Comprehensive integration test for the entire WTS Track 5 codebase.
Verifies all cross-module APIs match without requiring GPU or diffusers.
"""
import torch

# ─── Test 1: All imports ────────────────────────────────────────────
from models.tiv2_pipeline import DualModelPipeline, ModelType
from prompt.prompt_builder import PromptBuilder
from prompt.coect_processor import CoECTProcessor
from prompt.prompt_relay import PromptRelayRouter, PromptRelayHook, compute_segment_boundaries
from inference.ensemble import EnsembleGenerator, CLIPScorer, HistoryConsistencyScorer, TemporalCoherenceScorer
from utils.motion_analyzer import MotionAnalyzer, MotionFeatures
print("[PASS] Test 1: All module imports succeeded")

# ─── Test 2: PromptBuilder API ──────────────────────────────────────
pb = PromptBuilder(use_coect=False, use_motion_enrichment=False)
pkg = pb.build("ped walks", "car brakes")
required_keys = ["combined_prompt", "relay_inputs", "negative_prompt", "global_prompt",
                 "local_prompts", "segment_lengths", "enhanced_ped_prompt", "enhanced_veh_prompt"]
for k in required_keys:
    assert k in pkg, f"Missing key: {k}"
print(f"[PASS] Test 2: PromptBuilder.build() returns all {len(required_keys)} required keys")

# ─── Test 3: Prompt Relay segment boundaries ────────────────────────
b = compute_segment_boundaries(33, [0.4, 0.6])
assert len(b) == 2, f"Expected 2 segments, got {len(b)}"
latent_frames = ((33 - 1) // 4) + 1  # = 9
assert b[-1][1] == latent_frames - 1, f"Last frame should be {latent_frames-1}, got {b[-1][1]}"
print(f"[PASS] Test 3: Segment boundaries: {b}")

# ─── Test 4: MotionFeatures ─────────────────────────────────────────
mf = MotionFeatures(mean_flow=(1.0, 2.0), mean_speed=5.0, mean_direction_deg=90.0, acceleration=0.0)
d = mf.to_dict()
assert "speed_category" in d
assert "direction_category" in d
print(f"[PASS] Test 4: MotionFeatures - speed={d['speed_category']}, dir={d['direction_category']}")

# ─── Test 5: Scorers ────────────────────────────────────────────────
hs = HistoryConsistencyScorer()
ts = TemporalCoherenceScorer()
h_frames = torch.randn(5, 3, 64, 64)
g_frames = torch.randn(10, 3, 64, 64)
score_h = hs.score(h_frames, g_frames)
score_t = ts.score(g_frames)
assert 0.0 <= score_h <= 1.0, f"HistoryConsistency out of range: {score_h}"
assert 0.0 <= score_t <= 1.0, f"TemporalCoherence out of range: {score_t}"
print(f"[PASS] Test 5: Scorers - HistConsistency={score_h:.3f}, TempCoherence={score_t:.3f}")

# ─── Test 6: DualModelPipeline API ──────────────────────────────────
assert hasattr(DualModelPipeline, "generate_variable_n"), "Missing generate_variable_n"
assert hasattr(DualModelPipeline, "generate_with_latent_inpainting"), "Missing generate_with_latent_inpainting"
print("[PASS] Test 6: DualModelPipeline has generate_variable_n + generate_with_latent_inpainting")

# ─── Test 7: CoECT constructor modes ────────────────────────────────
# Test that "none" backend doesn't crash (no API key needed)
coect_none = CoECTProcessor(llm_backend="none")
assert coect_none._client is None
print("[PASS] Test 7: CoECTProcessor(llm_backend='none') works without API keys")

# ─── Test 8: Cross-check EnsembleGenerator expects correct pipeline API
import inspect
sig = inspect.signature(DualModelPipeline.generate_variable_n)
params = list(sig.parameters.keys())
assert "history_frames" in params
assert "prompt" in params
assert "n_frames" in params
assert "seed" in params
print(f"[PASS] Test 8: generate_variable_n signature matches EnsembleGenerator calls")

print()
print("=" * 50)
print("ALL 8 INTEGRATION TESTS PASSED")
print("=" * 50)
