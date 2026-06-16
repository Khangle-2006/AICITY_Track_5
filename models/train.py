"""
src/models/train.py
===================
LoRA Fine-Tuning for Generative Traffic Video Forecasting

This script fine-tunes Wan2.2 TI2V-5B using the WTS dataset.
Crucially, it utilizes the Latent Inpainting Mask strategy (aligned with
arXiv:2601.03665 Masked Latent Alignment) to train the model to
reconstruct the FUTURE trajectory while conditioning on the UNMASKED HISTORY.
"""

import os
import torch
import torch.nn.functional as F
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.optimization import get_scheduler

# Import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise ImportError("Please install peft: pip install peft")

def train_loop(
    dataset,
    output_dir: str = "checkpoints/wts-lora",
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    epochs: int = 10,
    batch_size: int = 1,
    lr: float = 1e-4,
):
    device = "cuda"
    dtype = torch.bfloat16

    print(f"Loading base model {model_id}...")
    # Separate VAE loading for Wan2.2 (float32 required)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype).to(device)
    
    # Freeze everything except LoRA
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Inject LoRA into the transformer's attention and feedforward layers
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.print_trainable_parameters()

    optimizer = torch.optim.AdamW(pipe.transformer.parameters(), lr=lr)
    
    # Simple dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting LoRA Fine-Tuning with Latent Inpainting Mask...")
    pipe.transformer.train()

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            # batch contains:
            # - history_frames: (B, K, C, H, W)
            # - target_frames: (B, N, C, H, W)
            # - caption_pedestrian: List[str]
            # - caption_vehicle: List[str]
            history_frames = batch["history_frames"].to(device, dtype)
            future_frames = batch["target_frames"].to(device, dtype)
            
            # Reconstruct prompts
            prompts = []
            for ped_cap, veh_cap in zip(batch["caption_pedestrian"], batch["caption_vehicle"]):
                prompt = f"A traffic scene where {veh_cap.lower()} while {ped_cap.lower()}"
                prompts.append(prompt)

            B = history_frames.shape[0]

            # 1. Encode full ground truth video (history + future)
            full_video = torch.cat([history_frames, future_frames], dim=1)
            # VAE expects (B, C, T, H, W)
            full_video = full_video.permute(0, 2, 1, 3, 4)
            with torch.no_grad():
                latents = pipe.vae.encode(full_video).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            # K_latent is the latent length of the history frames
            K_latent = (history_frames.shape[1] - 1) // 4 + 1

            # 2. Sample random timesteps (Flow Matching uses uniform t in [0, 1])
            # Wan2.2 uses FlowMatchEulerDiscreteScheduler
            timesteps = torch.randint(0, 1000, (B,), device=device).long()
            
            # 3. Add noise to latents (Flow-Matching forward process)
            noise = torch.randn_like(latents)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # 4. Create Mask (0 for history, 1 for future)
            # This is the core of the Masked Latent Alignment
            mask = torch.ones_like(latents)
            mask[:, :, :K_latent, :, :] = 0.0

            # 5. Force history to be the exact noisy ground truth
            # The network only needs to denoise the future (mask == 1)
            model_input = latents * (1 - mask) + noisy_latents * mask

            # 6. Get text embeddings robustly
            with torch.no_grad():
                prompt_outputs = pipe._encode_prompt(
                    prompt=prompts,
                    device=device,
                    num_videos_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                if isinstance(prompt_outputs, tuple):
                    prompt_embeds = prompt_outputs[0]
                else:
                    prompt_embeds = prompt_outputs

            # 7. Predict noise/velocity
            transformer_output = pipe.transformer(
                hidden_states=model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
            )
            if hasattr(transformer_output, "sample"):
                noise_pred = transformer_output.sample
            elif isinstance(transformer_output, tuple):
                noise_pred = transformer_output[0]
            else:
                noise_pred = transformer_output

            # 8. Compute Flow-Matching Loss ONLY on the masked future regions
            # In Flow-Matching, the model predicts the velocity field (v_t = x_1 - x_0)
            # where x_1 is the noise and x_0 is the data (latents).
            target = noise - latents
            
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
            
            # Apply Temporal Masking! 
            # We ONLY penalize the model for getting the future wrong. 
            loss = (loss * mask).sum() / mask.sum()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

    print("Training complete. Saving LoRA...")
    pipe.transformer.save_pretrained(output_dir)

if __name__ == "__main__":
    print("Run this script with accelerate for distributed training.")

