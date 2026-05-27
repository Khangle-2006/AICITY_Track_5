"""
Temporal In-Context Fine-Tuning (TIC-FT) Buffer Implementation

Based on: "Temporal In-Context Fine-Tuning with Temporal Reasoning for 
Versatile Control of Video Diffusion Models" (arxiv 2506.00996)

Key idea:
1. Concatenate condition frames (history) and target frames along temporal axis
2. Insert intermediate buffer frames with progressively increasing noise levels
3. These buffers smooth transitions and align with pretrained model's temporal dynamics
4. Enables smooth multi-chunk generation with minimal discontinuities
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class TICFTBufferManager:
    """
    Manages TIC-FT buffer frame insertion for smooth rolling-window inference.
    
    When generating long video sequences (e.g., 81 frames) with chunked diffusion:
    
    Chunk 1 (frames 1-16):
        [History (5 frames)] → [Buffer (5 noisy frames)] → [Generated (16 frames)]
    
    Chunk 2 (frames 17-32):
        [Last 5 generated from Chunk 1] → [Buffer (5 noisy)] → [Generated (16 frames)]
    
    The buffer frames provide smooth transitions and prevent temporal artifacts.
    """
    
    def __init__(
        self,
        noise_schedule: str = "linear",
        buffer_frames: int = 5,
        chunk_size: int = 16,
        overlap_frames: int = 4,
    ):
        """
        Args:
            noise_schedule: How to schedule noise in buffer frames ("linear" or "cosine")
            buffer_frames: Number of intermediate buffer frames between history and generation
            chunk_size: Number of frames to generate per chunk
            overlap_frames: Number of frames to overlap between chunks for smoothing
        """
        self.noise_schedule = noise_schedule
        self.buffer_frames = buffer_frames
        self.chunk_size = chunk_size
        self.overlap_frames = overlap_frames
    
    def get_noise_schedule(self, num_frames: int, timesteps: int = 1000) -> torch.Tensor:
        """
        Generate noise level schedule for buffer frames.
        
        Args:
            num_frames: Number of buffer frames
            timesteps: Max timestep value (default 1000 for DDPM)
        
        Returns:
            Tensor of shape (num_frames,) with noise timesteps
        """
        if self.noise_schedule == "linear":
            # Linear increase: 0 → timesteps
            schedule = torch.linspace(0, timesteps - 1, num_frames, dtype=torch.long)
        elif self.noise_schedule == "cosine":
            # Cosine schedule (smoother transitions)
            t = torch.linspace(0, np.pi / 2, num_frames)
            schedule = (torch.cos(t) * (timesteps - 1)).long()
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")
        
        return schedule
    
    def insert_buffer_frames(
        self,
        condition_frames: torch.Tensor,
        target_frames: torch.Tensor,
        buffer_noise_levels: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Insert noise-increasing buffer frames between condition and target.
        
        Args:
            condition_frames: Input history [B, T_cond, C, H, W] (in [0,1])
            target_frames: Target generation frames [B, T_tgt, C, H, W] (in [0,1])
            buffer_noise_levels: Optional noise schedule for buffers
            device: Device to place tensors on
        
        Returns:
            (concatenated_frames, buffer_noise_levels, buffer_mask)
            - concatenated: [B, T_cond + T_buf + T_tgt, C, H, W]
            - buffer_noise_levels: [T_buf] noise schedule
            - buffer_mask: [B, T_cond + T_buf + T_tgt] mask indicating buffer positions
        """
        if device is None:
            device = condition_frames.device
        
        B, _, C, H, W = condition_frames.shape
        T_cond = condition_frames.shape[1]
        T_tgt = target_frames.shape[1] if target_frames is not None else 0
        
        # Generate buffer noise schedule if not provided
        if buffer_noise_levels is None:
            buffer_noise_levels = self.get_noise_schedule(self.buffer_frames)
        
        buffer_noise_levels = buffer_noise_levels.to(device)
        
        # Create buffer frames: start from last condition frame, add noise progressively
        buffer_frames = self._create_buffer_frames(
            condition_frames[:, -1:],  # Start from last condition frame
            num_buffers=self.buffer_frames,
            noise_schedule=buffer_noise_levels,
            device=device,
        )  # [B, T_buf, C, H, W]
        
        # Concatenate: condition → buffer → target
        if target_frames is not None:
            concatenated = torch.cat([condition_frames, buffer_frames, target_frames], dim=1)
        else:
            concatenated = torch.cat([condition_frames, buffer_frames], dim=1)
        
        # Create mask: 0 for condition/target, 1 for buffer
        T_total = concatenated.shape[1]
        buffer_mask = torch.zeros(B, T_total, device=device, dtype=torch.bool)
        buffer_mask[:, T_cond:T_cond + self.buffer_frames] = True
        
        return concatenated, buffer_noise_levels, buffer_mask
    
    def _create_buffer_frames(
        self,
        seed_frame: torch.Tensor,
        num_buffers: int,
        noise_schedule: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create intermediate buffer frames with progressively increasing noise.
        
        Buffer frames are generated by adding noise to the seed frame,
        with noise level controlled by noise_schedule.
        
        Args:
            seed_frame: Starting frame [B, 1, C, H, W]
            num_buffers: Number of buffer frames to create
            noise_schedule: Noise levels [num_buffers]
            device: Device to place tensors on
        
        Returns:
            Buffer frames [B, num_buffers, C, H, W]
        """
        B, _, C, H, W = seed_frame.shape
        
        # Normalize seed to [-1, 1] if in [0, 1]
        seed = seed_frame.clone()
        if seed.min() >= 0:
            seed = seed * 2.0 - 1.0
        
        buffers = []
        for t in noise_schedule:
            # Convert timestep to noise level [0, 1]
            noise_level = float(t) / 1000.0
            
            # Add Gaussian noise scaled by noise_level
            noise = torch.randn_like(seed)
            noisy_frame = seed + noise_level * noise
            noisy_frame = torch.clamp(noisy_frame, -1.0, 1.0)
            
            buffers.append(noisy_frame)
        
        # Stack all buffers
        buffer_frames = torch.cat(buffers, dim=1)  # [B, num_buffers, C, H, W]
        
        return buffer_frames
    
    def get_chunk_indices(
        self,
        total_frames: int,
        start_frame: int = 0,
    ) -> List[Tuple[int, int]]:
        """
        Calculate frame indices for rolling-window chunked generation.
        
        For 81 total frames with chunk_size=16 and overlap_frames=4:
        
        Chunk 1: frames 0-15   (condition: 0-4, buffer: 5-9, generate: 10-15)
        Chunk 2: frames 12-27  (condition: 12-15, buffer: 16-20, generate: 21-27)
        ...
        
        Args:
            total_frames: Total frames to generate
            start_frame: Starting frame index
        
        Returns:
            List of (start, end) tuples for each chunk
        """
        chunks = []
        current_pos = start_frame
        
        while current_pos < total_frames:
            chunk_end = min(current_pos + self.chunk_size, total_frames)
            chunks.append((current_pos, chunk_end))
            
            # Move to next chunk position (with overlap for smoothing)
            current_pos += self.chunk_size - self.overlap_frames
        
        return chunks
    
    def blend_chunk_boundaries(
        self,
        prev_chunk_frames: torch.Tensor,
        curr_chunk_frames: torch.Tensor,
        overlap_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Blend overlapping frames between chunks to reduce discontinuities.
        
        Uses linear interpolation: output = α * prev + (1-α) * curr
        
        Args:
            prev_chunk_frames: Frames from previous chunk [B, T, C, H, W]
            curr_chunk_frames: Frames from current chunk [B, T, C, H, W]
            overlap_frames: Number of overlapping frames (default: self.overlap_frames)
        
        Returns:
            Blended frames for the overlap region [B, overlap_frames, C, H, W]
        """
        if overlap_frames is None:
            overlap_frames = self.overlap_frames
        
        # Take last overlap_frames from prev, first overlap_frames from curr
        prev_overlap = prev_chunk_frames[:, -overlap_frames:]  # [B, overlap, C, H, W]
        curr_overlap = curr_chunk_frames[:, :overlap_frames]    # [B, overlap, C, H, W]
        
        # Linear blend: more weight to curr as we progress through overlap
        alphas = torch.linspace(0, 1, overlap_frames, device=prev_chunk_frames.device)
        alphas = alphas.view(1, -1, 1, 1, 1)  # [1, overlap, 1, 1, 1]
        
        blended = (1 - alphas) * prev_overlap + alphas * curr_overlap
        
        return blended


class RollingWindowInferenceEngine:
    """
    Manages rolling-window multi-chunk video generation with TIC-FT buffers.
    """
    
    def __init__(
        self,
        pipeline,
        tic_ft_manager: TICFTBufferManager,
        device: torch.device = None,
    ):
        self.pipeline = pipeline
        self.tic_ft_manager = tic_ft_manager
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def generate_long_video(
        self,
        history_frames: torch.Tensor,
        prompt_embeds: torch.Tensor,
        total_frames: int,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        **pipeline_kwargs,
    ) -> torch.Tensor:
        """
        Generate a long video sequence using rolling-window chunks with TIC-FT.
        
        Args:
            history_frames: Condition frames [B, T_hist, C, H, W]
            prompt_embeds: Pre-computed text embeddings
            total_frames: Total frames to generate
            negative_prompt_embeds: Optional negative embeddings for CFG
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of diffusion steps per chunk
            **pipeline_kwargs: Additional kwargs for pipeline
        
        Returns:
            Generated video frames [B, total_frames, C, H, W]
        """
        B, T_hist, C, H, W = history_frames.shape
        
        # Initialize output with history
        output_frames = history_frames.clone()
        
        # Calculate how many targets we generate per step
        T_buf = self.tic_ft_manager.buffer_frames
        chunk_size = self.tic_ft_manager.chunk_size
        num_targets = chunk_size - T_hist - T_buf
        
        if num_targets <= 0:
            raise ValueError(f"Invalid parameters: chunk_size ({chunk_size}) must be greater than T_hist ({T_hist}) + T_buf ({T_buf})")
            
        print(f"🎬 Generating {total_frames} frames using autoregressive TIC-FT loop")
        print(f"   Chunk size: {chunk_size}, Buffer frames: {T_buf}, Targets per step: {num_targets}")
        
        step = 0
        target_length = total_frames
        
        while output_frames.shape[1] < T_hist + target_length:
            step += 1
            print(f"\n📍 Step {step}: frames {output_frames.shape[1]} to {min(output_frames.shape[1] + num_targets, T_hist + target_length)}")
            
            # Get condition frames: last T_hist frames from output
            condition = output_frames[:, -T_hist:].clone()
            
            # Create progressive noise buffer frames
            buffer_frames = self.tic_ft_manager._create_buffer_frames(
                seed_frame=condition[:, -1:],
                num_buffers=T_buf,
                noise_schedule=self.tic_ft_manager.get_noise_schedule(T_buf).to(condition.device),
                device=condition.device,
            )
            # Convert buffer frames from [-1, 1] to [0, 1] for pipeline input
            buffer_frames_01 = (buffer_frames * 0.5 + 0.5).clamp(0.0, 1.0)
            
            # Target placeholders: copy the last buffer frame or condition frame
            targets = condition[:, -1:].repeat(1, num_targets, 1, 1, 1)
            
            # Concatenate condition + buffer + targets
            input_video = torch.cat([condition, buffer_frames_01, targets], dim=1)
            
            # Generate target frames for this chunk
            target_frames = self.pipeline(
                video=input_video,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=H,
                width=W,
                output_type="pt",
                **pipeline_kwargs,
            ).frames  # [B, chunk_size, C, H, W]
            
            # Extract only the last num_targets generated frames
            new_generated = target_frames[:, -num_targets:]
            new_generated = new_generated.to(device=output_frames.device, dtype=output_frames.dtype)
            
            # Append generated frames to output
            output_frames = torch.cat([output_frames, new_generated], dim=1)
            print(f"   ✓ Generated chunk, total frames so far: {output_frames.shape[1]}")
            
        # Trim to exact requested length (history + total_frames)
        return output_frames[:, :T_hist + target_length]
