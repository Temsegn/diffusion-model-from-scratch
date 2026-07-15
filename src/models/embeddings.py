"""
Sinusoidal timestep embeddings for DDPM.

---------------------------------------------------------------------------
Why timestep embedding is required
---------------------------------------------------------------------------
The same U-Net must denoise at every noise level t ∈ {0,…,T-1}.
At t≈0 the residual is tiny high-frequency noise; at t≈999 the image is
nearly pure Gaussian. Injecting a learned function of t into every ResBlock
tells the network *which* denoising behavior to apply.

---------------------------------------------------------------------------
Formula (Transformer / DDPM style sinusoidal encoding)
---------------------------------------------------------------------------
For dimension index i = 0 … half_dim - 1:

    emb(t)[2i]   = sin( t / 10000^{2i / d} )
    emb(t)[2i+1] = cos( t / 10000^{2i / d} )

Then an MLP lifts this to `time_emb_dim` for ResBlock injection.

Shapes:
  t:          (B,)
  sinusoidal: (B, dim)
  after MLP:  (B, time_emb_dim)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Fixed sinusoidal features of the integer timestep, then MLP projection.

    Args:
        dim: Output embedding dimension (time_emb_dim), must be even for simplicity.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"time embedding dim must be even, got {dim}")
        self.dim = dim

        # MLP: (B, dim) → (B, dim) → (B, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Long / float tensor of shape (B,)

        Returns:
            Time embedding of shape (B, dim)
        """
        half = self.dim // 2
        # freqs: (half,) — 1 / 10000^{i/half}
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        # t[:, None] * freqs[None, :] → (B, half)
        args = t.float()[:, None] * freqs[None, :]
        # concat sin | cos → (B, dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)
