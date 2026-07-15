"""
Checkpoint save / load utilities for DDPM training.

Saved payload matches the internship presentation contract:

    {
        "epoch": int,
        "model": state_dict,
        "optimizer": state_dict,
        "loss": float,
    }

Files written at milestone epochs:

    checkpoints/ddpm_epoch_10.pt
    checkpoints/ddpm_epoch_50.pt
    checkpoints/ddpm_epoch_100.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a training checkpoint.

    Args:
        path: Destination file path (created if parent dirs missing).
        model: U-Net (or any nn.Module).
        optimizer: Optimizer whose state_dict is stored.
        epoch: Epoch number just completed (1-indexed end-of-epoch).
        loss: Average training loss for that epoch.
        extra: Optional extra keys (e.g. config dict, loss_history).

    Returns:
        Path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": float(loss),
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint into model (and optionally optimizer).

    Returns:
        The full checkpoint dictionary.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def checkpoint_path_for_epoch(checkpoint_dir: Union[str, Path], epoch: int) -> Path:
    """Return `checkpoints/ddpm_epoch_{epoch}.pt`."""
    return Path(checkpoint_dir) / f"ddpm_epoch_{epoch}.pt"
