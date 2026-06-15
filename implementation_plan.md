# Generative Traffic Video Forecasting (WTS Track 5) - SOTA Pipeline Plan

Based on your thorough analysis of the challenge, the limitations of current models, and the provided state-of-the-art repositories and papers, here is the proposed ultimate pipeline to achieve Top-1 on the leaderboard and write a high-impact paper.

## 1. Architectural Decision: The Wan2.2 + Wan-Animate Hybrid Strategy

You have raised a crucial point: **Wan-Animate does not allow text prompt inputs**, which makes it impossible to directly use the provided pedestrian and vehicle captions. 

**Our Decision**: We will **NOT** use the Wan-Animate model directly. Instead, we will use **Wan2.2 TI2V-5B (Text-Image-to-Video)** as our core generative engine, but we will heavily borrow the *reasoning and latent conditioning techniques* from Wan-Animate.

Why TI2V-5B?
* It takes **both** an Image (for visual consistency) and Text (for behavioral control) as first-class inputs.
* We solve the "history frames" conditioning by creating a **Motion-Aware History Composite**. Instead of just passing the *last* frame, we temporally blend the history frames into a single "motion ghost" image (similar to Wan-Animate's reference frame). This gives the model explicit spatial and velocity context.

## 2. Handling Variable History and Future Frames

The challenge states that history frames and future frames ($N$) are not fixed.
* **Variable History**: The `Motion-Aware History Composite` dynamically takes the last $K$ frames (e.g., $K=5$). If $K$ is small, it blends fewer; if $K$ is large, it takes the most recent $K$ to encode the trajectory.
* **Variable Future ($N$)**: Wan2.2 TI2V natively generates up to 121 frames. 
  * If $N \le 121$: We generate valid VAE frame counts and trim to exactly $N$.
  * If $N > 121$: We use **Chunked Autoregressive Generation with Overlap Blending**. We generate the first 121 frames, take the last 8 frames as the new "history", and generate the next chunk, seamlessly blending the overlaps in latent space using a cosine cross-fade.

## 3. Integrating the SOTA Toolchain

We will integrate the repositories and papers you linked to create an unbeatable pipeline:

1. **CoECT (Causal Event Cognition Theory)**: We will use this to parse the dual long-form captions (pedestrian & vehicle) into a structured timeline (Pre-recognition $\rightarrow$ Action $\rightarrow$ Avoidance).
2. **Prompt-Relay**: Instead of feeding a massive text prompt at frame 0, we use Prompt-Relay to inject specific parts of the parsed CoECT captions at specific temporal chunks of the video generation. This ensures the vehicle doesn't "brake" before the "action" phase.
3. **Cache-DiT & TeaCache (arxiv:2411.18977)**: We will implement timestep caching to bypass DiT block computations when latent differences are small. This provides a 3x-5x speedup, allowing us to generate an **Ensemble** of videos and pick the best one (using CLIP-S scoring) within the challenge time limits.
4. **Advanced Latent Alignment (arxiv:2601.03665 / 2603.14482)**: We will apply Masked Latent Alignment and Latent Rescaling during the VAE decoding step to preserve the high-frequency details of the original history frames, maximizing PSNR and SSIM metrics.

## 4. Proposed Changes

We will systematically update and verify the following components:

### `tiv2_pipeline.py`
- [MODIFY] Ensure the `MotionAwareComposite` optimally implements the "ghosting" technique to perfectly replicate the Wan-Animate conditioning effect.
- [MODIFY] Finalize the `_generate_chunked` method for seamless autoregressive generation for long $N$.

### `coect_processor.py` & `prompt_relay.py`
- [MODIFY] Connect `coect_processor` to parse the JSON captions into temporal stages.
- [MODIFY] Connect `prompt_relay` to the diffusers pipeline to temporally route these text prompts to the Cross-Attention layers.

### `ensemble.py` (New/Modify)
- [NEW] Implement an automated scoring script that generates $M$ candidates (using the speedup from Cache-DiT) and selects the best one based on an internal CLIP-S and temporal consistency metric.

## User Review Required

> [!IMPORTANT]
> **Do you agree with the strategic pivot away from the Wan-Animate model towards using Wan2.2 TI2V-5B with a Motion-Aware Composite?** This allows us to use text prompts while keeping the temporal consistency of Wan-Animate.

> [!TIP]
> **Ensemble Generation**: Since we are using Cache-DiT to speed up inference, we have extra compute time. Do you approve building an ensemble selector that generates 3 variations per segment and automatically picks the one with the highest visual fidelity (to maximize PSNR/SSIM/FVD)?

Please review this plan. Once approved, I will implement and test the remaining code pieces, ensuring everything is heavily commented for the technical report and paper.
