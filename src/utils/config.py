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
    # Save a checkpoint every N epochs (1 = save after every epoch).
    # Set to 0 to disable interval saving and use only save_epochs.
    save_every: int = 1
    # Extra milestone epochs (always saved even if not on the save_every grid)
    save_epochs: List[int] = field(default_factory=lambda: [10, 50, 100])

    # --- Sampling / clarity grids ---
    # Folder for generated sample PNGs (clarity improves across epochs).
    sample_dir: str = "./samples"
    # How many images to generate for a clarity grid.
    num_samples: int = 16
    # Save a sample grid every N epochs (0 = only use sample_epochs).
    sample_every: int = 0
    # Milestone epochs for clarity comparison (e.g. 10 blurry → 100 clearer).
    sample_epochs: List[int] = field(default_factory=lambda: [10, 50, 100])
    # Clip predicted x0 during reverse sampling for sharper / clearer images.
    clip_denoised: bool = True
