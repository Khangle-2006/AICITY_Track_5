"""
run_inference.py
================
End-to-End Inference Script for WTS Track 5

Runs the full pipeline:
1. Load pipeline.yaml config
2. CoECT Prompt Decomposition
3. DualModelPipeline with Bernini Latent Inpainting + Physics Conditioning
4. TeaCache + CacheDiT acceleration
5. Ensemble Generation (K candidates) -> Selection
"""

import sys
import os
import yaml
import torch

from models.tiv2_pipeline import DualModelPipeline
from prompt.prompt_builder import PromptBuilder
from inference.ensemble import EnsembleGenerator


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Load master config ──────────────────────────────────────────
    config_path = os.path.join(os.path.dirname(__file__), "configs", "pipeline.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("1. Initializing SOTA Pipeline (Wan2.2 + Bernini + Physics)...")
    pipeline = DualModelPipeline(config=config, device=device)

    # ── 2. Ensemble Generator ──────────────────────────────────────────
    ensemble_cfg = config.get("ensemble", {})
    print("2. Initializing Ensemble Generator...")
    ensemble = EnsembleGenerator(
        pipeline=pipeline,
        n_candidates=ensemble_cfg.get("n_candidates", 5),
        base_seed=ensemble_cfg.get("base_seed", 42),
        clip_weight=ensemble_cfg.get("weights", {}).get("clip_s", 0.40),
        hist_weight=ensemble_cfg.get("weights", {}).get("hist_consistency", 0.30),
        temp_weight=ensemble_cfg.get("weights", {}).get("ssim_anomaly", 0.20),
        vjepa_weight=ensemble_cfg.get("weights", {}).get("vjepa_sim", 0.10),
        enable_vjepa=False,  # Set True if V-JEPA 2 checkpoint is available
    )

    # ── 3. Build Prompt ────────────────────────────────────────────────
    coect_cfg = config.get("coect", {})
    motion_cfg = config.get("motion", {})
    prompt_builder = PromptBuilder(
        use_coect=coect_cfg.get("enabled", True),
        use_motion_enrichment=motion_cfg.get("enabled", True),
        coect_llm_backend=coect_cfg.get("llm_backend", "none"),
        coect_llm_model=coect_cfg.get("llm_model", "claude-sonnet-4-6"),
        flow_method=motion_cfg.get("flow_method", "farneback"),
        flow_device=motion_cfg.get("flow_device", "cpu"),
        coect_cache_dir=coect_cfg.get("cache_dir", "./cache/coect"),
    )

    print("3. Processing WTS Sample...")
    # Demo: create dummy history frames (replace with real data in production)
    history_frames = torch.randn((8, 3, 480, 720))

    ped_cap = (
        "A pedestrian is waiting at the curb, pre-recognizing the incoming traffic, "
        "then takes action to cross the street."
    )
    veh_cap = (
        "A blue sedan approaches the intersection, recognizes the pedestrian, "
        "and brakes to avoid collision."
    )

    prompt_pkg = prompt_builder.build(ped_cap, veh_cap, history_frames=history_frames)
    final_prompt = prompt_pkg["combined_prompt"]
    print(f"Final Prompt:\n{final_prompt}")

    # ── 4. Generate ────────────────────────────────────────────────────
    gen_cfg = config.get("generation", {})
    print("4. Generating Future Sequence...")
    n_frames = gen_cfg.get("default_num_frames", 33)
    if n_frames <= 0:
        n_frames = 33
        
    best_frames, info = ensemble.generate_and_select(
        history_frames=history_frames,
        prompt=final_prompt,
        n_frames=n_frames,
        generation_kwargs={
            "guidance_scale": gen_cfg.get("guidance_scale", 5.0),
            "num_inference_steps": gen_cfg.get("num_inference_steps", 50),
        },
        prompt_package=prompt_pkg,
    )

    print(f"5. Done! Best Candidate Index: {info['best_idx']}")
    print(f"Output shape: {best_frames.shape}")


if __name__ == "__main__":
    main()
