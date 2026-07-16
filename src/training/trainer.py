

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
from src.training.checkpoint import (
    backup_training_artifacts,
    checkpoint_path_for_epoch,
    save_checkpoint,
    save_loss_history,
)
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
            f"lr={self.cfg.learning_rate} T={self.cfg.timesteps} "
            f"save_every={self.cfg.save_every}"
        )
        if self.cfg.backup_dir:
            print(f"[Trainer] backup_dir={self.cfg.backup_dir}")

    def _backup_artifacts(self) -> None:
        """Copy checkpoints to backup_dir if configured."""
        if not self.cfg.backup_dir:
            return
        backup_path = backup_training_artifacts(
            checkpoint_dir=self.checkpoint_dir,
            backup_dir=self.cfg.backup_dir,
            loss_history=self.loss_history,
        )
        print(f"  Backed up artifacts → {backup_path}")

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

    def _should_save_epoch(self, epoch: int) -> bool:
        """True when this epoch should write a checkpoint."""
        if self.cfg.save_every and self.cfg.save_every > 0:
            if epoch % self.cfg.save_every == 0:
                return True
        return epoch in self.cfg.save_epochs

    def train(self) -> List[float]:
        """
        Full training over `cfg.epochs`.

        Saves checkpoints every `cfg.save_every` epochs (default: every epoch)
        and at any extra milestones in `cfg.save_epochs`.

        Returns:
            loss_history — one float per completed epoch
        """
        for epoch in range(1, self.cfg.epochs + 1):
            avg_loss = self.train_one_epoch(epoch)
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch}/{self.cfg.epochs} — avg loss: {avg_loss:.6f}")

            if self._should_save_epoch(epoch):
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
                self._backup_artifacts()

        save_loss_history(self.checkpoint_dir, self.loss_history)
        self._backup_artifacts()
        return self.loss_history
