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

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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


def save_loss_history(
    checkpoint_dir: Union[str, Path],
    loss_history: List[float],
) -> Path:
    """Save per-epoch losses as JSON next to checkpoints."""
    path = Path(checkpoint_dir) / "loss_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epochs": list(range(1, len(loss_history) + 1)),
        "loss_history": [float(x) for x in loss_history],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def backup_training_artifacts(
    checkpoint_dir: Union[str, Path],
    backup_dir: Union[str, Path],
    loss_history: Optional[List[float]] = None,
    sample_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Copy checkpoints, loss history, and clarity sample grids to a backup folder.

    Useful on Google Colab to persist models to Google Drive:

        cfg.backup_dir = "/content/drive/MyDrive/ddpm-checkpoints"
    """
    checkpoint_dir = Path(checkpoint_dir)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for ckpt in sorted(checkpoint_dir.glob("ddpm_epoch_*.pt")):
        shutil.copy2(ckpt, backup_dir / ckpt.name)
        copied += 1

    loss_path = checkpoint_dir / "loss_history.json"
    if loss_history is not None:
        save_loss_history(checkpoint_dir, loss_history)
    if loss_path.exists():
        shutil.copy2(loss_path, backup_dir / loss_path.name)

    if sample_dir is not None:
        sample_dir = Path(sample_dir)
        sample_backup = backup_dir / "samples"
        sample_backup.mkdir(parents=True, exist_ok=True)
        for png in sorted(sample_dir.glob("samples_epoch_*.png")):
            shutil.copy2(png, sample_backup / png.name)
            copied += 1

    if copied == 0 and not (backup_dir / "loss_history.json").exists():
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir} to back up."
        )

    return backup_dir
