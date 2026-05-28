"""
Wan2.2 V2V Pipeline.
Customizes HF WanVideoToVideoPipeline with Inference-time Latent Flow Guidance.
"""

import math
import sys
import torch
import torch.nn.functional as F

import diffusers
from diffusers import WanVideoToVideoPipeline as HFWanPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput


class Wan2_2V2VPipeline(HFWanPipeline):
    """
    Subclass of HF WanVideoToVideoPipeline with Inference-time Latent Flow Guidance and Semantic Expert Routing.
    """

    _optional_components = ["transformer", "transformer_2"]

    def __init__(
        self,
        tokenizer,
        text_encoder,
        transformer,
        vae,
        scheduler,
        transformer_2=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )
        self.register_modules(transformer_2=transformer_2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def latent_flow_guidance_step(self, latent_pred, history_flow_map, lambda_weight=0.1):
        """
        Compute optical flow from predicted latent, compute L2 loss against history flow map,
        and return the updated latent prediction corrected by the gradient.
        """
        with torch.enable_grad():
            latent_pred_for_grad = latent_pred.detach().requires_grad_(True)
            latent_gray = latent_pred_for_grad.mean(dim=1)

            Ix = (torch.roll(latent_gray, shifts=-1, dims=3) - torch.roll(latent_gray, shifts=1, dims=3)) / 2.0
            Iy = (torch.roll(latent_gray, shifts=-1, dims=2) - torch.roll(latent_gray, shifts=1, dims=2)) / 2.0
            It = torch.roll(latent_gray, shifts=-1, dims=1) - latent_gray

            Ix = Ix[:, :-1]
            Iy = Iy[:, :-1]
            It = It[:, :-1]

            eps = 1e-4
            den = Ix**2 + Iy**2 + eps
            u = -It * Ix / den
            v = -It * Iy / den

            flow_pred = torch.stack([u, v], dim=2).mean(dim=1)

            B, C_flow, H_lat, W_lat = flow_pred.shape
            if history_flow_map.shape[-2:] != (H_lat, W_lat):
                history_flow_map_resized = F.interpolate(
                    history_flow_map, size=(H_lat, W_lat), mode="bilinear", align_corners=False
                ) / 8.0
            else:
                history_flow_map_resized = history_flow_map

            history_flow_map_resized = history_flow_map_resized.to(device=flow_pred.device, dtype=flow_pred.dtype)
            loss = F.mse_loss(flow_pred, history_flow_map_resized)

            gradient = torch.autograd.grad(loss, latent_pred_for_grad)[0]
            gradient = torch.nan_to_num(gradient, nan=0.0, posinf=100.0, neginf=-100.0)
            grad_norm = gradient.norm()
            if grad_norm > 100.0:
                gradient = gradient * (100.0 / grad_norm)

        return latent_pred - lambda_weight * gradient

    @torch.no_grad()
    def __call__(
        self,
        video=None,
        prompt=None,
        negative_prompt=None,
        height=480,
        width=832,
        num_inference_steps=50,
        timesteps=None,
        guidance_scale=5.0,
        strength=0.8,
        num_videos_per_prompt=None,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="np",
        return_dict=True,
        attention_kwargs=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
        min_guidance_scale=1.0,
        use_dynamic_cfg=True,
        history_flow_map=None,
        lambda_weight=0.1,
        phase_label=None,
        guidance_scale_2=None,
    ):
        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1

        self.check_inputs(
            prompt, negative_prompt, height, width, video, latents,
            prompt_embeds, negative_prompt_embeds, callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.transformer.device
        transformer_dtype = self.transformer.dtype

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        prompt_embeds = prompt_embeds.to(device=device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=transformer_dtype)

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, timesteps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        self._num_timesteps = len(timesteps)

        if latents is None:
            video_tensor = self.video_processor.preprocess_video(video, height=height, width=width).to(
                device, dtype=self.vae.dtype
            )
        else:
            video_tensor = None

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            video_tensor,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            self.vae.dtype,
            device,
            generator,
            latents,
            latent_timestep,
        )

        if guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        boundary_timestep = None
        if hasattr(self, "transformer_2") and self.transformer_2 is not None:
            boundary_ratio = 0.875
            if phase_label is not None:
                p_val = int(phase_label) if not isinstance(phase_label, torch.Tensor) else int(phase_label.item())
                if p_val in [0, 1]:
                    boundary_ratio = 0.50
                elif p_val in [3, 4]:
                    boundary_ratio = 0.95
                elif p_val == 2:
                    boundary_ratio = 0.875
            else:
                boundary_ratio = getattr(self.config, "boundary_ratio", 0.875)
                if boundary_ratio is None:
                    boundary_ratio = 0.875

            num_train_timesteps = getattr(self.scheduler.config, "num_train_timesteps", 1000)
            boundary_timestep = boundary_ratio * num_train_timesteps

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        def is_on_gpu(module):
            try:
                return next(module.parameters()).device.type == "cuda"
            except StopIteration:
                return False

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                if self.do_classifier_free_guidance:
                    if use_dynamic_cfg:
                        progress = i / max(len(timesteps) - 1, 1)
                        step_guidance_scale = min_guidance_scale + (guidance_scale - min_guidance_scale) * (
                            0.5 * (1.0 + math.cos(math.pi * progress))
                        )
                        step_guidance_scale_2 = min_guidance_scale + (guidance_scale_2 - min_guidance_scale) * (
                            0.5 * (1.0 + math.cos(math.pi * progress))
                        )
                    else:
                        step_guidance_scale = guidance_scale
                        step_guidance_scale_2 = guidance_scale_2

                t_val = t.item() if isinstance(t, torch.Tensor) else float(t)

                # Determine which expert to run and hot-swap if necessary
                use_transformer_2 = False
                if boundary_timestep is not None and hasattr(self, "transformer_2") and self.transformer_2 is not None:
                    if t_val < boundary_timestep:
                        use_transformer_2 = True

                if use_transformer_2:
                    if not is_on_gpu(self.transformer_2):
                        print(f"🔄 Hot-swapping at t={t_val:.1f}: Moving transformer to CPU and transformer_2 to {device}...")
                        self.transformer = self.transformer.to("cpu")
                        torch.cuda.empty_cache()
                        self.transformer_2 = self.transformer_2.to(device)
                    active_transformer = self.transformer_2
                    active_guidance_scale = step_guidance_scale_2 if self.do_classifier_free_guidance else guidance_scale_2
                else:
                    if not is_on_gpu(self.transformer):
                        print(f"🔄 Hot-swapping at t={t_val:.1f}: Moving transformer_2 to CPU and transformer to {device}...")
                        if hasattr(self, "transformer_2") and self.transformer_2 is not None:
                            self.transformer_2 = self.transformer_2.to("cpu")
                        torch.cuda.empty_cache()
                        self.transformer = self.transformer.to(device)
                    active_transformer = self.transformer
                    active_guidance_scale = step_guidance_scale if self.do_classifier_free_guidance else guidance_scale

                # Compute active expert prediction
                with active_transformer.cache_context("cond") if hasattr(active_transformer, "cache_context") else torch.no_grad():
                    noise_pred = active_transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with active_transformer.cache_context("uncond") if hasattr(active_transformer, "cache_context") else torch.no_grad():
                        noise_uncond = active_transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + active_guidance_scale * (noise_pred - noise_uncond)

                if history_flow_map is not None and lambda_weight > 0.0:
                    sigma = t_val / 1000.0
                    latent_pred = latents - sigma * noise_pred
                    latent_pred_guided = self.latent_flow_guidance_step(
                        latent_pred, history_flow_map, lambda_weight=lambda_weight
                    )
                    noise_pred = (latents - latent_pred_guided) / max(sigma, 1e-6)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)


diffusers.Wan2_2V2VPipeline = Wan2_2V2VPipeline
sys.modules["diffusers"].Wan2_2V2VPipeline = Wan2_2V2VPipeline
