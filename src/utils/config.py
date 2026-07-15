"""
Training and diffusion hyperparameters.

Why this exists:
  Keeps DDPM paper defaults (T=1000, linear beta schedule, AdamW lr) in one place
  so Trainer, Scheduler, notebook, and CLI all share the same configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    """Hyperparameters for CIFAR-10 DDPM training."""

    # --- Dataset ---
    image_size: int = 32
    channels: int = 3
    data_root: str = "./data"

    # --- Diffusion schedule (DDPM paper defaults) ---
    # T = number of forward noising steps
    timesteps: int = 1000
    # Linear beta schedule: β_1 = 1e-4 → β_T = 0.02
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # --- Training ---
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 2e-4
    num_workers: int = 2  # use 0 on Windows if DataLoader spawn fails
    seed: int = 42

    # --- U-Net (small CIFAR-scale) ---
    base_channels: int = 64
    channel_mults: tuple = (1, 2, 2)  # 32 → 16 → 8 spatial resolutions
    num_res_blocks: int = 2
    time_emb_dim: int = 256

    # --- Checkpoints ---
    checkpoint_dir: str = "./checkpoints"
    # Optional persistent backup (e.g. Google Drive on Colab).
    # When set, checkpoints and loss history are copied here after each save
    # and again at the end of training.
    backup_dir: Optional[str] = None
    # Save full checkpoint at these epoch numbers (1-indexed end of epoch)
    save_epochs: List[int] = field(default_factory=lambda: [10, 50, 100])
