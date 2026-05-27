import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

class LoRALinear(nn.Module):
    """
    LoRA-adapted Linear layer.
    Given Wx, we add α * B(A(x)).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=bias)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.lora_a.weight, std=1 / math.sqrt(self.rank))
        nn.init.zeros_(self.lora_b.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_out = self.lora_b(self.lora_a(x))
        if self.dropout is not None:
            lora_out = self.dropout(lora_out)
        return self.alpha * lora_out

class LoRAWrappedLinear(nn.Module):
    """
    Wraps an existing nn.Linear layer and adds a parallel LoRALinear path.
    """
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALinear(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=False
        )
        self._is_enabled = False
        
        # Freeze the base linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

    def enable(self):
        self._is_enabled = True
        for param in self.lora.parameters():
            param.requires_grad = True

    def disable(self):
        self._is_enabled = False
        for param in self.lora.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast input to match the wrapped linear layer dtype
        x_casted = x.to(dtype=self.linear.weight.dtype)
        out = self.linear(x_casted)
        
        if self._is_enabled:
            # Cast back to lora parameter dtype if necessary
            lora_out = self.lora(x.to(dtype=self.lora.lora_a.weight.dtype))
            out = out + lora_out.to(dtype=out.dtype)
            
        return out

class WAN2_2TemporalLoRAAdapter(nn.Module):
    """
    Complete LoRA adapter for WAN2.2 transformer joint attention layers.
    Freezes base model parameters and adds trainable LoRA adapters on attn1/attn2 linear projections.
    """
    def __init__(
        self,
        transformer: nn.Module,
        vae: nn.Module,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.transformer = transformer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lora_modules: nn.ModuleList = nn.ModuleList()
        
        # Freeze base models completely before applying adapters
        self.transformer.requires_grad_(False)
        vae.requires_grad_(False)
        
        self._apply_temporal_lora()
        
    def _apply_temporal_lora(self):
        """Scan transformer and apply LoRA directly to projection linear layers in attn1/attn2."""
        for name, module in self.transformer.named_modules():
            # Detect joint attention layers
            if ('attn1' in name.lower() or 'attn2' in name.lower()) and hasattr(module, 'to_q'):
                # Wrap to_q, to_k, to_v
                module.to_q = LoRAWrappedLinear(module.to_q, self.rank, self.alpha, self.dropout)
                self.lora_modules.append(module.to_q)
                
                module.to_k = LoRAWrappedLinear(module.to_k, self.rank, self.alpha, self.dropout)
                self.lora_modules.append(module.to_k)
                
                module.to_v = LoRAWrappedLinear(module.to_v, self.rank, self.alpha, self.dropout)
                self.lora_modules.append(module.to_v)
                
                # Wrap output projection
                if isinstance(module.to_out, (nn.Sequential, nn.ModuleList)):
                    module.to_out[0] = LoRAWrappedLinear(module.to_out[0], self.rank, self.alpha, self.dropout)
                    self.lora_modules.append(module.to_out[0])
                else:
                    module.to_out = LoRAWrappedLinear(module.to_out, self.rank, self.alpha, self.dropout)
                    self.lora_modules.append(module.to_out)
                    
    def enable_lora(self):
        for m in self.lora_modules:
            m.enable()
            
    def disable_lora(self):
        for m in self.lora_modules:
            m.disable()
            
    def get_trainable_parameters(self) -> int:
        total = 0
        for m in self.lora_modules:
            for p in m.lora.parameters():
                if p.requires_grad:
                    total += p.numel()
        return total
        
    def save_lora_weights(self, checkpoint_path: str):
        """Save only LoRA adapter weights."""
        full_state_dict = self.lora_modules.state_dict()
        lora_state_dict = {k: v for k, v in full_state_dict.items() if 'lora.' in k}
        torch.save(lora_state_dict, checkpoint_path)
        print(f"✓ Saved LoRA weights to {checkpoint_path}")
        
    def load_lora_weights(self, checkpoint_path: str, device: torch.device = None):
        """Load LoRA adapter weights."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # Filter keys to make sure we only load lora weights
        state_dict = {k: v for k, v in state_dict.items() if 'lora.' in k}
        self.lora_modules.load_state_dict(state_dict, strict=False)
        if device:
            self.lora_modules.to(device)
        print(f"✓ Loaded LoRA weights from {checkpoint_path}")

    def forward(self, *args, **kwargs):
        """Pass through to transformer."""
        return self.transformer(*args, **kwargs)

    @property
    def dtype(self) -> torch.dtype:
        return self.transformer.dtype

    @property
    def device(self) -> torch.device:
        return self.transformer.device

    @property
    def config(self):
        return self.transformer.config

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "transformer" in self.__dict__:
                return getattr(self.transformer, name)
            raise

def apply_lora_to_pipeline(
    pipeline,
    rank: int = 8,
    alpha: float = 1.0,
    enable_training: bool = False,
) -> WAN2_2TemporalLoRAAdapter:
    adapter = WAN2_2TemporalLoRAAdapter(
        transformer=pipeline.transformer,
        vae=pipeline.vae,
        rank=rank,
        alpha=alpha,
        dropout=0.05,
    )
    if enable_training:
        adapter.enable_lora()
    else:
        adapter.disable_lora()
        
    # Replace transformer in pipeline with adapter wrapper
    pipeline.transformer = adapter
    
    print(f"✓ Applied LoRA adapter to transformer")
    print(f"  - Trainable parameters: {adapter.get_trainable_parameters():,}")
    print(f"  - LoRA rank: {rank}")
    print(f"  - Temporal adapters: {len(adapter.lora_modules)}")
    
    return adapter
