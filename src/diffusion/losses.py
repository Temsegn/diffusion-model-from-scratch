"""
DDPM training objective — noise prediction MSE loss.

---------------------------------------------------------------------------
Why DDPM predicts noise ε instead of the clean image x_0
---------------------------------------------------------------------------
The variational lower bound (ELBO) on log p(x_0) can be rewritten so that
each term asks the model to match the true posterior mean μ̃_t(x_t, x_0).

Using the reparameterization

    x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε

that posterior mean is an *affine function of ε*. Predicting ε̂_θ(x_t, t)
is therefore equivalent (up to a time-dependent scaling that DDPM drops)
to predicting μ̃_t — but with a simpler, more stable target:

    ε ~ N(0, I)               # same scale for every t
    L_simple = E_{t,x_0,ε} [ || ε - ε̂_θ(x_t, t) ||² ]

Predicting x_0 directly is possible but often less stable at large t, where
almost all signal is gone and the network would chase a near-impossible
reconstruction from pure noise.

---------------------------------------------------------------------------
Shapes
---------------------------------------------------------------------------
  x0, x_t, ε, ε̂ : (B, 3, 32, 32)
  t             : (B,)
  loss          : scalar
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.diffusion.forward import add_noise
from src.diffusion.scheduler import NoiseScheduler


def ddpm_loss(
    model: nn.Module,
    x0: torch.Tensor,
    scheduler: NoiseScheduler,
    t: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute one-batch DDPM simplified objective.

    Steps:
      1. Sample t ~ Uniform({0,...,T-1}) if not provided
      2. Sample ε ~ N(0, I)
      3. Form x_t = √(ᾱ_t) x0 + √(1-ᾱ_t) ε
      4. Predict ε̂ = model(x_t, t)
      5. loss = MSE(ε̂, ε)

    Args:
        model: U-Net mapping (x_t, t) → ε̂
        x0: Clean batch (B, C, H, W) in [-1, 1]
        scheduler: NoiseScheduler
        t: Optional pre-sampled timesteps (B,)

    Returns:
        loss: Scalar MSE loss
        t: Timesteps used (useful for logging)
    """
    batch_size = x0.shape[0]
    device = x0.device

    if t is None:
        t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device)

    # x_t, epsilon: both (B, C, H, W)
    x_t, epsilon = add_noise(x0, t, scheduler)

    # predicted_noise: (B, C, H, W)
    predicted_noise = model(x_t, t)

    loss = F.mse_loss(predicted_noise, epsilon)
    return loss, t
