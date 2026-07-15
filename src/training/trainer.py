"""
DDPM Trainer — full training loop for CIFAR-10.

Training configuration (defaults from TrainConfig):
  Dataset:      CIFAR-10
  Epochs:       100
  Batch size:   128
  Optimizer:    AdamW
  Learning rate: 2e-4
  Timesteps:    1000

Per-batch algorithm:
  1. Load clean images x0
  2. Sample random timestep t
  3. Add noise → (x_t, ε)
  4. Predict noise with U-Net
  5. MSE(predicted, ε)
  6. zero_grad → backward → optimizer.step
  7. Log loss; save checkpoint at configured epochs
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from torch.optim import AdamW
from tqdm import tqdm

from src.datasets.cifar10 import get_cifar10_dataloader
from src.diffusion.losses import ddpm_loss
from src.diffusion.scheduler import NoiseScheduler
from src.models.unet import build_unet
from src.training.checkpoint import checkpoint_path_for_epoch, save_checkpoint
from src.utils.config import TrainConfig
from src.utils.device import get_device
from src.utils.seed import set_seed


class Trainer:
    """
    End-to-end DDPM trainer.

    Attributes:
        cfg: TrainConfig
        loss_history: Per-epoch average MSE losses (for Colab plotting)
        device: torch.device
    """

    def __init__(self, cfg: Optional[TrainConfig] = None) -> None:
        self.cfg = cfg or TrainConfig()
        set_seed(self.cfg.seed)
        self.device = get_device()

        self.scheduler = NoiseScheduler.from_config(self.cfg, device=self.device)
        self.model = build_unet(self.cfg).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

        self.train_loader = get_cifar10_dataloader(
            batch_size=self.cfg.batch_size,
            data_root=self.cfg.data_root,
            num_workers=self.cfg.num_workers,
            train=True,
        )

        self.loss_history: List[float] = []
        self.checkpoint_dir = Path(self.cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Trainer] device={self.device}")
        print(f"[Trainer] params={sum(p.numel() for p in self.model.parameters()):,}")
        print(
            f"[Trainer] epochs={self.cfg.epochs} batch={self.cfg.batch_size} "
            f"lr={self.cfg.learning_rate} T={self.cfg.timesteps}"
        )

    def train_one_epoch(self, epoch: int) -> float:
        """Run one epoch; return mean batch loss."""
        self.model.train()
        running = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.cfg.epochs}",
            leave=True,
        )
        for images, _labels in pbar:
            # images: (B, 3, 32, 32) in [-1, 1]
            x0 = images.to(self.device)

            loss, _t = ddpm_loss(self.model, x0, self.scheduler)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            running += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running / max(n_batches, 1)

    def train(self) -> List[float]:
        """
        Full training over `cfg.epochs`.

        Saves checkpoints at epochs listed in `cfg.save_epochs`
        (default: 10, 50, 100).

        Returns:
            loss_history — one float per completed epoch
        """
        for epoch in range(1, self.cfg.epochs + 1):
            avg_loss = self.train_one_epoch(epoch)
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch}/{self.cfg.epochs} — avg loss: {avg_loss:.6f}")

            if epoch in self.cfg.save_epochs:
                path = checkpoint_path_for_epoch(self.checkpoint_dir, epoch)
                save_checkpoint(
                    path=path,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=avg_loss,
                    extra={"loss_history": list(self.loss_history)},
                )
                print(f"  Saved checkpoint → {path}")

        return self.loss_history
