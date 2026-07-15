"""
CIFAR-10 dataset loading for DDPM training.

---------------------------------------------------------------------------
Why normalization to [-1, 1] is required for diffusion models
---------------------------------------------------------------------------
Raw CIFAR pixels after ToTensor() live in [0, 1].
The forward process adds Gaussian noise ε ~ N(0, I), whose typical magnitude
is order ~1. If images stay in [0, 1], early noise dominates the signal scale
awkwardly; mapping to [-1, 1] puts signal and noise on a comparable scale,
which is the standard DDPM / Score-SDE convention:

    x̃_0 = (x_0 - 0.5) / 0.5 = 2 * x_0 - 1 ∈ [-1, 1]^{3×32×32}

---------------------------------------------------------------------------
Tensor shapes
---------------------------------------------------------------------------
  Single image:  (3, 32, 32)       — C, H, W
  DataLoader batch: (B, 3, 32, 32)
  Labels (unused in unconditional DDPM): (B,)

Dataset size: 50_000 training images, 10 classes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_transforms() -> transforms.Compose:
    """
    Build the transform pipeline for CIFAR-10.

    Steps:
      1. ToTensor()  — PIL / ndarray → float tensor in [0, 1], shape (3, 32, 32)
      2. Normalize(mean=0.5, std=0.5) on each channel → approximately [-1, 1]
         Formula:  (x - mean) / std  =  (x - 0.5) / 0.5
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """
    Map tensors from approximately [-1, 1] back to [0, 1] for visualization.

    Args:
        x: Tensor shaped (..., 3, H, W) or (3, H, W)

    Returns:
        Tensor in [0, 1] (clamped).
    """
    # Inverse of Normalize(0.5, 0.5): x = x * 0.5 + 0.5
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def get_cifar10_dataloader(
    batch_size: int = 128,
    data_root: str = "./data",
    num_workers: int = 2,
    train: bool = True,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Create a CIFAR-10 DataLoader (auto-downloads on first run).

    Args:
        batch_size: Mini-batch size B.
        data_root: Folder where torchvision stores/downloads CIFAR-10.
        num_workers: DataLoader workers (set 0 on Windows if needed).
        train: If True, use the 50_000-image training split.
        shuffle: Defaults to True for train, False for test.

    Returns:
        DataLoader yielding (images, labels):
          images: (B, 3, 32, 32) float32 in [-1, 1]
          labels: (B,) int64  — ignored by unconditional DDPM
    """
    if shuffle is None:
        shuffle = train

    dataset = datasets.CIFAR10(
        root=data_root,
        train=train,
        download=True,
        transform=get_cifar10_transforms(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,  # keep batch size fixed during training
    )


def get_cifar10_dataloaders(
    batch_size: int = 128,
    data_root: str = "./data",
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Convenience: return (train_loader, test_loader)."""
    train_loader = get_cifar10_dataloader(
        batch_size=batch_size,
        data_root=data_root,
        num_workers=num_workers,
        train=True,
    )
    test_loader = get_cifar10_dataloader(
        batch_size=batch_size,
        data_root=data_root,
        num_workers=num_workers,
        train=False,
    )
    return train_loader, test_loader
