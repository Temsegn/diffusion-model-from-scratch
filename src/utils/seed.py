"""
Reproducibility helpers.

Why this exists:
  Diffusion training samples random timesteps and Gaussian noise every batch.
  Fixing seeds makes runs comparable when debugging or demoing for a presentation.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN can be slower; enable when you need exact reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
