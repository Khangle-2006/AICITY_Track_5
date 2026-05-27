"""
SOTA Video-to-Video Pipeline with Wan2.1 Core for AI City Challenge 2026 Track 5

Architecture (per docx spec):
  1. Wan2.1 3D Causal VAE for spatio-temporal compression
  2. WanTransformer3D (Rectified Flow DiT) for foreground generation
  3. UMT5-XXL text encoder for dual/mono text conditioning
  4. Optical flow background disentanglement (homography warping)
  5. Adaptive alpha blending for foreground-background compositing
  6. Dynamic CFG scheduling
  7. Autoregressive sliding window for long sequences
  8. Step-caching for temporal context preservation

This modular pipeline navigates the evaluation trilemma:
  - Maximizes PSNR/SSIM via deterministic background preservation
  - Maximizes LPIPS/FVD via Wan2.1 flow-matching DiT
  - Maximizes CLIP-S via UMT5 text conditioning
"""

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from diffusers import (
    AutoencoderKLWan,
    WanTransformer3DModel,
    WanVideoToVideoPipeline as HFWanPipeline,
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import is_torch_xla_available
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    xm = None
    XLA_AVAILABLE = False

from src.datasets.transforms import denormalize_image
from src.models.disentanglement.optical_flow import BackgroundPreservationModule
from src.models.compositing.alpha_blending import ForegroundBackgroundCompositor

PHASE_NAMES = {
    0: "pre-recognition",
    1: "recognition",
    2: "judgment",
    3: "action",
    4: "avoidance/outcome",
}

DOMAIN_NAMES = {
    "bdd_front": "BDD front-view driving camera",
    "wts_vehicle": "WTS vehicle or IP camera",
    "wts_overhead": "WTS overhead fixed camera",
}


@dataclass
class WanPipelineConfig:
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    torch_dtype: str = "bfloat16"
    height: int = 480
    width: int = 832
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    min_guidance: float = 1.0
    strength: float = 0.7
    flow_shift: float = 3.0
    max_sequence_length: int = 512
    chunk_size: int = 81
    output_width: int = 1280
    output_height: int = 720
    use_background_preservation: bool = True
    use_alpha_compositing: bool = True
    use_dynamic_cfg: bool = True
    use_step_caching: bool = True
    use_token_pruning: bool = True
    prune_threshold: float = 0.1
    overlap_frames: int = 2
    ransac_threshold: float = 1.0
    motion_threshold: float = 0.05
    flow_stride: int = 4
    negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, subtitles, "
        "worst quality, low quality, JPEG compression residue, ugly, incomplete, "
        "deformed, disfigured, misshapen, still picture, messy background"
    )


def build_wan_prompt(caption: str, phase_label: int, sample_type: str) -> str:
    phase_name = PHASE_NAMES.get(int(phase_label), f"phase {int(phase_label)}")
    domain_name = DOMAIN_NAMES.get(sample_type, sample_type or "unknown camera")
    return (
        f"Traffic video forecasting, {domain_name}, future {phase_name} phase, "
        f"highly detailed, high resolution, clear road layout, realistic motion, sharp focus. "
        f"Preserve camera viewpoint, road layout, actors, motion continuity, lighting, and scale. "
        f"{caption}"
    )


class DynamicCFGScheduler:
    """Dynamic CFG scheduling per docx Section 7.2.
    
    High CFG at initial steps for macro-structure and motion trajectories.
    Decaying CFG at later steps for high-frequency detail refinement.
    """
    def __init__(
        self,
        max_guidance: float = 7.5,
        min_guidance: float = 1.0,
        schedule_type: str = "cosine",
    ):
        self.max_guidance = max_guidance
        self.min_guidance = min_guidance
        self.schedule_type = schedule_type

    def get_guidance(self, step: int, total_steps: int) -> float:
        progress = step / max(total_steps - 1, 1)
        if self.schedule_type == "linear":
            return self.max_guidance - (self.max_guidance - self.min_guidance) * progress
        elif self.schedule_type == "cosine":
            return self.min_guidance + (self.max_guidance - self.min_guidance) * (
                0.5 * (1 + np.cos(np.pi * progress))
            )
        elif self.schedule_type == "exponential":
            return self.min_guidance + (self.max_guidance - self.min_guidance) * (
                np.exp(-5 * progress)
            )
        return self.max_guidance


class DynamicCFGWanVideoToVideoPipeline(HFWanPipeline):
    @torch.no_grad()
    def __call__(
        self,
        video: list[Image.Image] = None,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        timesteps: list[int] | None = None,
        guidance_scale: float = 5.0,
        strength: float = 0.8,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        min_guidance_scale: float = 1.0,
        use_dynamic_cfg: bool = True,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            video,
            latents,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.transformer.device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt on CPU to save GPU VRAM
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=torch.device("cpu"),
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=transformer_dtype)

        # 4. Prepare timesteps
        if XLA_AVAILABLE:
            timestep_device = "cpu"
        else:
            timestep_device = device
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, timestep_device, timesteps
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, timesteps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        self._num_timesteps = len(timesteps)

        if latents is None:
            video = self.video_processor.preprocess_video(video, height=height, width=width).to(
                device, dtype=torch.float32
            )

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            video,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
            latent_timestep,
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                    # Step-level dynamic CFG decay
                    if use_dynamic_cfg:
                        progress = i / max(len(timesteps) - 1, 1)
                        # Cosine schedule
                        step_guidance_scale = min_guidance_scale + (guidance_scale - min_guidance_scale) * (
                            0.5 * (1.0 + math.cos(math.pi * progress))
                        )
                    else:
                        step_guidance_scale = guidance_scale

                    noise_pred = noise_uncond + step_guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

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

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)


class WanSOTAPipeline:
    """
    Full SOTA pipeline with Wan2.1 core, background disentanglement,
    alpha compositing, dynamic CFG, and autoregressive chunking.
    """

    def __init__(
        self,
        config: WanPipelineConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_pipeline()
        self._init_aux_modules()
        self.cfg_scheduler = DynamicCFGScheduler(
            config.guidance_scale, config.min_guidance
        )
        self.step_cache = {}

    def _load_pipeline(self):
        dtype = getattr(torch, self.config.torch_dtype)
        model_id = self.config.model_id

        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = DynamicCFGWanVideoToVideoPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=dtype,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=self.config.flow_shift
        )
        # We manually load vae and transformer to GPU, and keep text_encoder on CPU to avoid GPU OOM.
        pipe.vae.to(self.device)
        pipe.transformer.to(self.device)
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8

    def _init_aux_modules(self):
        self.background_module = BackgroundPreservationModule(
            use_raft=False,
            ransac_threshold=self.config.ransac_threshold,
            motion_threshold=self.config.motion_threshold,
            flow_stride=self.config.flow_stride,
        )
        self.compositor = ForegroundBackgroundCompositor(
            use_depth_aware=False,
            motion_mask_weight=0.5,
        )

    def build_prompts(
        self,
        captions: Sequence[str],
        phase_labels: torch.Tensor,
        sample_types: Sequence[str],
    ) -> List[str]:
        labels = phase_labels.detach().cpu().tolist()
        return [
            build_wan_prompt(caption, label, sample_type)
            for caption, label, sample_type in zip(captions, labels, sample_types)
        ]

    def _prepare_video_for_wan(self, frames: torch.Tensor) -> torch.Tensor:
        """Convert [B, T, C, H, W] in [-1,1] -> [B, F, C, H, W] in [0,1] for WAN VideoProcessor."""
        return (frames * 0.5 + 0.5).clamp(0.0, 1.0)


    @torch.no_grad()
    def generate_batch(
        self,
        batch: Dict[str, Any],
        prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        history_frames = batch["history_frames"].to(self.device)
        phase_labels = batch.get(
            "phase_label",
            torch.zeros(history_frames.shape[0], dtype=torch.long, device=self.device),
        )
        target_frames = batch.get("frame_length", [self.config.chunk_size])
        if isinstance(target_frames, list):
            target_frames = target_frames[0]
        target_frames = int(target_frames)

        if prompt_embeds is not None:
            prompt_str = prompt or ""
        else:
            captions = batch.get("merged_caption", [""] * history_frames.shape[0])
            sample_types = batch.get("sample_type", ["unknown"] * history_frames.shape[0])
            prompts = self.build_prompts(captions, phase_labels, sample_types)
            prompt_str = prompt or prompts[0]

        B, T_hist, C, H, W = history_frames.shape
        chunk_size = self.config.chunk_size
        overlap_frames = self.config.overlap_frames

        # Initialize final output tensor
        final_frames = torch.zeros(B, target_frames, C, H, W, dtype=history_frames.dtype, device=self.device)

        # Generate first chunk
        C_1 = min(target_frames, chunk_size)
        chunk_1 = self._generate_chunk(
            history_frames, prompt_str, C_1, 0,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        n_1 = chunk_1.shape[1]
        final_frames[:, :n_1] = chunk_1
        L = n_1
        chunk_idx = 1

        # Generate subsequent chunks autoregressively
        while L < target_frames:
            # Overlap window size
            O = min(overlap_frames, L)
            t_start = L - O
            C_k = min(target_frames - t_start, chunk_size)

            # Retrieve history window ending at t_start
            temp_seq = torch.cat([history_frames, final_frames[:, :L]], dim=1)
            curr_history = temp_seq[:, t_start : t_start + T_hist]

            # Generate chunk k
            chunk_k = self._generate_chunk(
                curr_history, prompt_str, C_k, chunk_idx,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )
            n_k = chunk_k.shape[1]

            # Apply linear alpha blending on the overlap region
            O_eff = min(O, n_k)
            n_write = min(n_k, target_frames - t_start)
            if O_eff > 0 and n_write > O_eff:
                alpha = torch.linspace(0.0, 1.0, steps=O_eff, device=self.device).view(1, O_eff, 1, 1, 1).to(history_frames.dtype)
                blend = final_frames[:, t_start : t_start + O_eff] * (1.0 - alpha) + chunk_k[:, :O_eff] * alpha
                final_frames[:, t_start : t_start + O_eff] = blend
                final_frames[:, t_start + O_eff : t_start + n_write] = chunk_k[:, O_eff : n_write]
            elif n_write > 0:
                final_frames[:, t_start : t_start + n_write] = chunk_k[:, :n_write]

            L = t_start + n_write
            chunk_idx += 1

        return final_frames

    @torch.no_grad()
    def _generate_chunk(
        self,
        history_frames: torch.Tensor,
        prompt: str,
        num_frames: int,
        chunk_idx: int,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T_hist, C, H, W = history_frames.shape

        # Pad num_frames to avoid VAE temporal truncation:
        # VAE temporal factor is 4, so (T_hist + num_frames_pad - 1) must be divisible by 4
        vae_temp_factor = 4
        pad_extra = (vae_temp_factor - (T_hist + num_frames - 1) % vae_temp_factor) % vae_temp_factor
        num_frames_pad = num_frames + pad_extra

        if self.config.use_background_preservation:
            # Generate the future background for the entire target duration
            background, _ = self.background_module(history_frames, num_frames_pad)
            # Concatenate history frames and warped background to guide the video-to-video pipeline
            frames_input = torch.cat([history_frames, background], dim=1)
        else:
            # Fallback: simple repetition of the last history frame
            last_frame = history_frames[:, -1:]
            rep_frames = last_frame.repeat(1, num_frames_pad, 1, 1, 1)
            frames_input = torch.cat([history_frames, rep_frames], dim=1)

        vid_tensor = self._prepare_video_for_wan(frames_input)

        guidance = self.config.guidance_scale

        if prompt_embeds is not None:
            output = self.pipe(
                video=vid_tensor,
                prompt_embeds=prompt_embeds.to(self.device),
                negative_prompt_embeds=negative_prompt_embeds.to(self.device) if negative_prompt_embeds is not None else None,
                height=self.config.height,
                width=self.config.width,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=guidance,
                strength=self.config.strength,
                output_type="pt",
                return_dict=False,
                min_guidance_scale=self.config.min_guidance,
                use_dynamic_cfg=self.config.use_dynamic_cfg,
            )[0]
        else:
            output = self.pipe(
                video=vid_tensor,
                prompt=prompt,
                negative_prompt=self.config.negative_prompt,
                height=self.config.height,
                width=self.config.width,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=guidance,
                strength=self.config.strength,
                output_type="pt",
                return_dict=False,
                min_guidance_scale=self.config.min_guidance,
                use_dynamic_cfg=self.config.use_dynamic_cfg,
            )[0]

        # Slice off history frames and handle VAE temporal truncation
        avail_future = output.shape[1] - T_hist
        actual_n = min(num_frames, avail_future)
        future_generated = output[:, T_hist:T_hist + actual_n]

        # Map to [-1, 1] range
        chunk_frames = future_generated * 2.0 - 1.0
        chunk_frames = chunk_frames.clamp(-1, 1)

        if self.config.use_background_preservation:
            chunk_frames = self._apply_background_compositing(
                chunk_frames, history_frames
            )

        return chunk_frames

    @torch.no_grad()
    def _apply_background_compositing(
        self,
        generated_frames: torch.Tensor,
        history_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply background preservation + alpha compositing.
        
        Uses optical flow to estimate background motion via homography,
        then composites generated frames as foreground over warped background.
        """
        B, T_gen, C, H, W = generated_frames.shape

        background, motion_mask = self.background_module(
            history_frames, T_gen
        )

        foreground = generated_frames.clone()

        alpha_mask = torch.ones(B, 1, T_gen, H, W, device=self.device)

        background_bcthw = background.permute(0, 2, 1, 3, 4)
        foreground_bcthw = foreground.permute(0, 2, 1, 3, 4)
        alpha_bcthw = alpha_mask

        composited = self.compositor(
            foreground_bcthw,
            background_bcthw,
            alpha_bcthw,
            motion_mask=motion_mask.unsqueeze(2) if motion_mask.dim() == 4 else motion_mask,
        )

        result = composited.permute(0, 2, 1, 3, 4)
        return result

    def resize_for_submission(self, frames: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            frames,
            size=(self.config.output_height, self.config.output_width),
            mode="bilinear",
            align_corners=False,
        )

    def save_batch(
        self, batch: Dict[str, Any], pred_frames: torch.Tensor, output_dir: str
    ) -> str:
        scenario_name = batch["scenario_name"][0]
        sample_dir = os.path.join(output_dir, scenario_name)
        os.makedirs(sample_dir, exist_ok=True)

        resized = self.resize_for_submission(pred_frames[0])
        for i in range(resized.size(0)):
            torchvision.utils.save_image(
                denormalize_image(resized[i]),
                os.path.join(sample_dir, f"{i}.png"),
            )

        if "future_frames" in batch:
            gt_frames = denormalize_image(batch["future_frames"])[0]
            gt_resized = self.resize_for_submission(gt_frames)
            gt_dir = os.path.join(output_dir, f"{scenario_name}_gt")
            os.makedirs(gt_dir, exist_ok=True)
            for i in range(gt_resized.size(0)):
                torchvision.utils.save_image(
                    gt_resized[i], os.path.join(gt_dir, f"{i}.png")
                )

        return sample_dir
