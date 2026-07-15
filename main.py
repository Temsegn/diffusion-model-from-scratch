"""
Main CLI entry point for DDPM From Scratch.

Usage:
    python main.py train [--epochs N] [--batch-size N] [--lr FLOAT]

Training is wired in Phase 6. Other commands (sample / restore) are deferred.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when run as:
#   python main.py train
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DDPM From Scratch — training CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train DDPM on CIFAR-10")
    train_p.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    train_p.add_argument("--batch-size", type=int, default=None, help="Batch size")
    train_p.add_argument("--lr", type=float, default=None, help="AdamW learning rate")
    train_p.add_argument("--seed", type=int, default=None, help="Random seed")
    train_p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints",
    )
    train_p.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="Persistent backup folder (e.g. Google Drive path on Colab)",
    )
    train_p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset download/cache directory",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        # Imported lazily so scaffold commits do not require torch yet.
        from src.training.trainer import Trainer
        from src.utils.config import TrainConfig

        cfg = TrainConfig()
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.lr is not None:
            cfg.learning_rate = args.lr
        if args.seed is not None:
            cfg.seed = args.seed
        if args.checkpoint_dir is not None:
            cfg.checkpoint_dir = args.checkpoint_dir
        if args.backup_dir is not None:
            cfg.backup_dir = args.backup_dir
        if args.data_root is not None:
            cfg.data_root = args.data_root

        trainer = Trainer(cfg)
        trainer.train()
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
