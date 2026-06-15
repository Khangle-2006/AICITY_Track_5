"""
run_inference.py
================
End-to-End Inference Script for WTS Track 5

Runs the full pipeline:
1. Load WTS JSON
2. CoECT Prompt Decomposition
3. DualModelPipeline with Latent Inpainting
4. TeaCache + CacheDiT acceleration
5. Ensemble Generation (K candidates) -> Selection
"""

import sys
import os
import torch

from models.tiv2_pipeline import DualModelPipeline, ModelType
from prompt.prompt_builder import PromptBuilder
from inference.ensemble import EnsembleGenerator

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("1. Initializing SOTA Latent Inpainting Pipeline...")
    pipeline = DualModelPipeline(
        primary_model=ModelType.TI2V_5B,
        device=device,
        enable_tea_cache=True,
        tea_cache_thresh=0.25,
        enable_cache_dit=True,
        lora_path=None
    )

    print("2. Initializing Ensemble Generator...")
    ensemble = EnsembleGenerator(
        pipeline=pipeline,
        n_candidates=3,
        enable_vjepa=False
    )

    print("3. Processing WTS Sample...")
    history_frames = torch.randn((5, 3, 480, 720), device=device)
    
    prompt_builder = PromptBuilder(
        use_coect=False,             # Disable LLM call for local dry-run
        use_motion_enrichment=False, # No RAFT needed for demo
    )
    
    ped_cap = "A pedestrian is waiting at the curb, pre-recognizing the incoming traffic, then takes action to cross the street."
    veh_cap = "A blue sedan approaches the intersection, recognizes the pedestrian, and brakes to avoid collision."
    
    prompt_pkg = prompt_builder.build(ped_cap, veh_cap)
    final_prompt = prompt_pkg["combined_prompt"]

    print(f"Final Prompt:\n{final_prompt}")

    print("4. Generating Future Sequence...")
    best_frames, info = ensemble.generate_and_select(
        history_frames=history_frames,
        prompt=final_prompt,
        n_frames=45,
        generation_kwargs={
            "guidance_scale": 5.0,
            "num_inference_steps": 50,
        },
        prompt_package=prompt_pkg
    )

    print(f"5. Done! Best Candidate Index: {info['best_idx']}")
    print(f"Output shape: {best_frames.shape}")

if __name__ == "__main__":
    main()
