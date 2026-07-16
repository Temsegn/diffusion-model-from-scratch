"""
Save generated sample grids to visualize clarity over training.

Images are stored in [-1, 1] during training; we map them to [0, 1] for PNG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torchvision.utils import make_grid, save_image


def tensor_to_display(images: torch.Tensor) -> torch.Tensor:
    """Map model images from [-1, 1] → [0, 1] for saving."""
    return (images.clamp(-1.0, 1.0) + 1.0) * 0.5


def save_sample_grid(
    images: torch.Tensor,
    path: Union[str, Path],
    nrow: int = 4,
) -> Path:
    """
    Save a batch of generated images as one PNG grid.

    Args:
        images: (N, C, H, W) in [-1, 1].
        path: Destination .png path.
        nrow: Images per row in the grid.

    Returns:
        Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    display = tensor_to_display(images.detach().cpu())
    grid = make_grid(display, nrow=nrow, padding=2)
    save_image(grid, path)
    return path
