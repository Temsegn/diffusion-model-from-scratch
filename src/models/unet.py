"""
DDPM U-Net denoiser implemented from scratch (no Hugging Face Diffusers).

ASCII architecture (CIFAR-10, 32×32, channel_mults=(1,2,2)):

    Input x_t (B, 3, 32, 32) + t (B,)
            │
       [stem Conv 3→64]                     → (B, 64, 32, 32)   ── skip
            │
       [ResBlock×2 @ 64]                    → (B, 64, 32, 32)   ── skips
       [Downsample]                         → (B, 64, 16, 16)
            │
       [ResBlock×2 @ 128]                   → (B,128, 16, 16)   ── skips
       [Downsample]                         → (B,128,  8,  8)
            │
       [ResBlock×2 @ 128]                   → (B,128,  8,  8)   ── skips
            │
       [Mid ResBlock×2]                     → (B,128,  8,  8)
            │
       [concat skip + ResBlock×2] @ 8×8
       [Upsample]                           → 16×16
            │
       [concat skip + ResBlock×2] @ 16×16
       [Upsample]                           → 32×32
            │
       [concat skip + ResBlock×2] @ 32×32
       [concat stem + ResBlock]
       [GroupNorm + SiLU + Conv → 3]        → (B, 3, 32, 32) = ε̂

Why skip connections:
  Encoder activations preserve fine spatial detail that the decoder needs
  to reconstruct the noise field accurately.

Why residual blocks + time embedding:
  Each ResBlock injects a projected time vector so the residual adapts to t.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.embeddings import SinusoidalTimeEmbedding
from src.utils.config import TrainConfig


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 32 if channels >= 32 else max(1, channels // 4)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    """
    ResBlock: GroupNorm → SiLU → Conv → (+time) → GroupNorm → SiLU → Conv → +skip

    Shapes example (in_ch=64, out_ch=128, H=W=32):
      x:        (B, 64, 32, 32)
      time_emb: (B, time_dim)
      out:      (B,128, 32, 32)
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # (B, out_ch) → (B, out_ch, 1, 1) broadcast over spatial dims
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    """(B, C, H, W) → (B, C, H/2, W/2) via stride-2 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """(B, C, H, W) → (B, C, 2H, 2W) via nearest ×2 + conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net noise predictor ε̂_θ(x_t, t).

    Input:  x (B, 3, 32, 32), t (B,)
    Output: ε̂ (B, 3, 32, 32)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.time_mlp = SinusoidalTimeEmbedding(time_emb_dim)

        # Stem: (B, 3, 32, 32) → (B, base, 32, 32)
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ---------- Encoder ----------
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        # Track skip channel widths: stem + every ResBlock output (NOT downsambles)
        skip_channels: List[int] = [ch]

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(ch, out_ch, time_emb_dim))
                ch = out_ch
                skip_channels.append(ch)
            self.downs.append(blocks)
            if level < len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(None)

        # ---------- Mid ----------
        self.mid1 = ResidualBlock(ch, ch, time_emb_dim)
        self.mid2 = ResidualBlock(ch, ch, time_emb_dim)

        # ---------- Decoder ----------
        # For each level (deep → shallow): num_res_blocks concat-skips, then Upsample
        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop()
                blocks.append(ResidualBlock(ch + skip_ch, out_ch, time_emb_dim))
                ch = out_ch
            self.ups.append(blocks)
            if level > 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(None)

        # Final stem skip
        stem_skip_ch = skip_channels.pop()
        assert len(skip_channels) == 0
        self.final_res = ResidualBlock(ch + stem_skip_ch, base_channels, time_emb_dim)
        ch = base_channels

        self.out_norm = _group_norm(ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # time_emb: (B, time_emb_dim)
        time_emb = self.time_mlp(t)

        # stem → (B, 64, 32, 32)
        h = self.stem(x)
        skips: List[torch.Tensor] = [h]

        # Encoder — only ResBlock outputs go onto the skip stack
        for blocks, down in zip(self.downs, self.downsamples):
            for block in blocks:
                h = block(h, time_emb)
                skips.append(h)
            if down is not None:
                h = down(h)

        # Mid → (B, C_mid, 8, 8)
        h = self.mid1(h, time_emb)
        h = self.mid2(h, time_emb)

        # Decoder
        for blocks, up in zip(self.ups, self.upsamples):
            for block in blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, time_emb)
            if up is not None:
                h = up(h)

        # Final stem skip
        skip = skips.pop()
        h = torch.cat([h, skip], dim=1)
        h = self.final_res(h, time_emb)

        assert len(skips) == 0, f"unused skips left: {len(skips)}"

        # (B, 3, 32, 32)
        return self.out_conv(F.silu(self.out_norm(h)))


def build_unet(cfg: Optional[TrainConfig] = None) -> UNet:
    """Factory using TrainConfig defaults."""
    if cfg is None:
        cfg = TrainConfig()
    return UNet(
        in_channels=cfg.channels,
        out_channels=cfg.channels,
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        time_emb_dim=cfg.time_emb_dim,
    )
