"""
Forward diffusion — add noise to clean images in closed form.

---------------------------------------------------------------------------
Equation (DDPM closed-form forward process)
---------------------------------------------------------------------------
    x_t = √(ᾱ_t) · x_0  +  √(1 - ᾱ_t) · ε

Where:
  x_0  = clean image     shape (B, 3, 32, 32)
  ε    ~ N(0, I)         shape (B, 3, 32, 32)
  t    = timesteps         shape (B,)
  x_t  = noisy image       shape (B, 3, 32, 32)
  ᾱ_t  = alphas_cumprod[t]

---------------------------------------------------------------------------
Why Gaussian noise?
---------------------------------------------------------------------------
Gaussians compose: a chain of Gaussian transitions stays Gaussian.
That gives the closed form above so training never needs to unroll all T
steps. The network then learns to predict ε (or equivalently denoise).

This function returns both x_t and ε because the training loss is
MSE(predicted_noise, ε).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.diffusion.scheduler import NoiseScheduler


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Gather schedule values for a batch of timesteps and reshape for broadcast.

    Args:
        a: 1-D schedule tensor, shape (T,)
        t: Long tensor of timesteps, shape (B,)
        x_shape: Target image shape (B, C, H, W) so we expand to (B, 1, 1, 1)

    Returns:
        Tensor shape (B, 1, 1, 1) ready to multiply with images of shape x_shape.
    """
    batch_size = t.shape[0]
    # a[t] → (B,); then reshape to (B, 1, 1, 1) for broadcasting over C,H,W
    out = a.gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def add_noise(
    x0: torch.Tensor,
    t: torch.Tensor,
    scheduler: NoiseScheduler,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the closed-form forward diffusion step.

        x_t = √(ᾱ_t) · x_0  +  √(1 - ᾱ_t) · ε

    Args:
        x0: Clean images, shape (B, C, H, W), typically in [-1, 1].
        t:  Timesteps, shape (B,), dtype long, values in [0, T-1].
        scheduler: NoiseScheduler providing √(ᾱ) and √(1-ᾱ).
        noise: Optional pre-sampled ε. If None, sample N(0, I).

    Returns:
        x_t:     Noisy images, shape (B, C, H, W)
        epsilon: The noise that was added, shape (B, C, H, W)
    """
    if noise is None:
        noise = torch.randn_like(x0)

    # sqrt_ab: (B, 1, 1, 1), sqrt_1m_ab: (B, 1, 1, 1)
    sqrt_ab = _extract(scheduler.sqrt_alphas_cumprod, t, x0.shape)
    sqrt_1m_ab = _extract(scheduler.sqrt_one_minus_alphas_cumprod, t, x0.shape)

    # Broadcast multiply: (B,1,1,1) * (B,C,H,W) → (B,C,H,W)
    x_t = sqrt_ab * x0 + sqrt_1m_ab * noise
    return x_t, noise
