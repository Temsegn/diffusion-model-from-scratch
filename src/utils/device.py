"""
Device selection helper.

Why this exists:
  Training and inference must run on CUDA when available (Colab GPU) and
  fall back to CPU otherwise. Centralizing this avoids scattered device logic.
"""

from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Return the best available torch.device.

    Args:
        prefer_cuda: If True, use CUDA when torch.cuda.is_available().

    Returns:
        torch.device("cuda") or torch.device("cpu")
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
