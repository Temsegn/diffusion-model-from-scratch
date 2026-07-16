"""
DDPM reverse (sampling) process — generate images from noise.

---------------------------------------------------------------------------
One reverse step (Ho et al., 2020)
---------------------------------------------------------------------------
Given noisy x_t and predicted noise ε̂_θ(x_t, t):

    x̂_0 = (x_t - √(1 - ᾱ_t) · ε̂) / √(ᾱ_t)     # predicted clean image
    x̂_0 = clip(x̂_0, -1, 1)                     # clarity: keep pixels valid

Posterior mean toward x_{t-1}:

    μ̃_t = (√(ᾱ_{t-1}) · β_t / (1 - ᾱ_t)) · x̂_0
         + (√(α_t) · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)) · x_t

    x_{t-1} = μ̃_t + √(β̃_t) · z ,   z ~ N(0, I) if t > 0 else 0

Clipping x̂_0 to [-1, 1] (matching CIFAR normalization) reduces color blow-up
and yields sharper, clearer samples — especially early in training.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from src.diffusion.forward import _extract
from src.diffusion.scheduler import NoiseScheduler


@torch.no_grad()
def p_sample_step(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    scheduler: NoiseScheduler,
    clip_denoised: bool = True,
) -> torch.Tensor:
    """
    Single reverse step: x_t → x_{t-1}.

    Args:
        model: Noise-predicting U-Net.
        x_t: Current noisy images (B, C, H, W).
        t: Timestep tensor (B,) with the same integer t for the batch.
        scheduler: NoiseScheduler with reverse quantities.
        clip_denoised: If True, clamp predicted x0 to [-1, 1] for clearer images.

    Returns:
        x_{t-1} of shape (B, C, H, W).
    """
    betas = _extract(scheduler.betas, t, x_t.shape)
    alphas = _extract(scheduler.alphas, t, x_t.shape)
    alphas_cumprod = _extract(scheduler.alphas_cumprod, t, x_t.shape)
    alphas_cumprod_prev = _extract(scheduler.alphas_cumprod_prev, t, x_t.shape)
    sqrt_one_minus_ab = _extract(scheduler.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    posterior_variance = _extract(scheduler.posterior_variance, t, x_t.shape)

    predicted_noise = model(x_t, t)

    # Predict x0 from ε-parameterization
    pred_x0 = (x_t - sqrt_one_minus_ab * predicted_noise) / torch.sqrt(alphas_cumprod)
    if clip_denoised:
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

    # Posterior mean μ̃_t(x_t, x̂_0)
    coef_x0 = torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod)
    coef_xt = torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    model_mean = coef_x0 * pred_x0 + coef_xt * x_t

    noise = torch.randn_like(x_t)
    nonzero_mask = (t != 0).float().reshape(-1, *([1] * (x_t.ndim - 1)))
    return model_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise


@torch.no_grad()
def sample(
    model: nn.Module,
    scheduler: NoiseScheduler,
    shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    clip_denoised: bool = True,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Full reverse chain: x_T ~ N(0, I) → x_0 (clearer image).

    Args:
        model: Trained U-Net (caller should set model.eval()).
        scheduler: NoiseScheduler.
        shape: Output shape, e.g. (n, 3, 32, 32).
        device: Device for sampling.
        clip_denoised: Clamp predicted x0 each step for better clarity.
        show_progress: Show tqdm over T reverse steps.

    Returns:
        Generated images in [-1, 1], shape `shape`.
    """
    device = device or next(model.parameters()).device
    x = torch.randn(shape, device=device)

    timesteps = list(range(scheduler.timesteps))[::-1]
    iterator = (
        tqdm(timesteps, desc="Sampling", leave=False) if show_progress else timesteps
    )

    for t_int in iterator:
        t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
        x = p_sample_step(model, x, t, scheduler, clip_denoised=clip_denoised)

    return x.clamp(-1.0, 1.0)
