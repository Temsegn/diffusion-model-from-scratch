"""
DDPM noise scheduler — beta / alpha / alpha_bar schedules from scratch.

---------------------------------------------------------------------------
Mathematical definitions (Ho et al., 2020)
---------------------------------------------------------------------------
Forward diffusion one step:

    q(x_t | x_{t-1}) = N( √(1 - β_t) · x_{t-1} ,  β_t · I )

Define:

    α_t      = 1 - β_t
    ᾱ_t      = ∏_{s=1}^{t} α_s     (cumulative product, "alpha_bar")

Closed-form noising of a clean image x_0 in one shot:

    q(x_t | x_0) = N( √(ᾱ_t) · x_0 ,  (1 - ᾱ_t) · I )
    x_t          = √(ᾱ_t) · x_0  +  √(1 - ᾱ_t) · ε ,   ε ~ N(0, I)

---------------------------------------------------------------------------
Why β increases (linear 1e-4 → 0.02)
---------------------------------------------------------------------------
Small early β keeps structure; larger late β destroys signal until x_T ≈ noise.
A smooth increase makes each step a mild corruption that the U-Net can learn.

---------------------------------------------------------------------------
Why α and ᾱ are needed
---------------------------------------------------------------------------
α packs (1 - β) for nicer formulas.
ᾱ collapses the Markov chain into a closed form — no need to loop t steps
to obtain x_t during training (huge speedup).

Tensor shapes (T = 1000):
  betas, alphas, alphas_cumprod : (T,)
"""

from __future__ import annotations

from typing import Optional

import torch

from src.utils.config import TrainConfig


class NoiseScheduler:
    """
    Precomputes β_t, α_t, ᾱ_t for the forward / reverse process.

    Args:
        timesteps: Number of diffusion steps T (default 1000).
        beta_start: β_1 (default 1e-4).
        beta_end:   β_T (default 0.02).
        device: Optional device to place schedule tensors on.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: Optional[torch.device] = None,
    ) -> None:
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Linear beta schedule: shape (T,)
        # β_t goes from beta_start at t=0 to beta_end at t=T-1
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        # α_t = 1 - β_t
        alphas = 1.0 - betas

        # ᾱ_t = ∏_{s=1}^{t} α_s   (cumprod along the timestep axis)
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        if device is not None:
            betas = betas.to(device)
            alphas = alphas.to(device)
            alphas_cumprod = alphas_cumprod.to(device)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

        # Useful derived quantities for training / sampling later
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Reverse process: ᾱ_{t-1} with ᾱ_{-1} := 1 for the t=0 edge case
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32, device=alphas_cumprod.device), alphas_cumprod[:-1]]
        )
        # β̃_t = ((1 - ᾱ_{t-1}) / (1 - ᾱ_t)) · β_t  (posterior variance)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # Clamp for numerical stability at t=0 where variance is ~0
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)

        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_variance = posterior_variance
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(alphas_cumprod_prev)

    @classmethod
    def from_config(
        cls,
        cfg: TrainConfig,
        device: Optional[torch.device] = None,
    ) -> "NoiseScheduler":
        """Build scheduler from TrainConfig."""
        return cls(
            timesteps=cfg.timesteps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            device=device,
        )

    def to(self, device: torch.device) -> "NoiseScheduler":
        """Move all schedule tensors to device (in-place)."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device
        )
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_alphas_cumprod_prev = self.sqrt_alphas_cumprod_prev.to(device)
        return self

    def get_alphas_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """
        Gather ᾱ_t for a batch of timesteps.

        Args:
            t: Long tensor of shape (B,) with values in [0, T-1]

        Returns:
            Tensor of shape (B,) with ᾱ_t values.
        """
        return self.alphas_cumprod[t]
