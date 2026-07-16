
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
        description="DDPM From Scratch — train / sample CLI",
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
    train_p.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N epochs (default: 1 = every epoch)",
    )
    train_p.add_argument(
        "--sample-every",
        type=int,
        default=None,
        help="Save clarity sample grid every N epochs (0 = milestones only)",
    )
    train_p.add_argument(
        "--sample-dir",
        type=str,
        default=None,
        help="Directory for clarity sample PNGs",
    )
    train_p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of images in each clarity grid",
    )

    sample_p = sub.add_parser(
        "sample",
        help="Generate clearer images from a checkpoint (reverse diffusion)",
    )
    sample_p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to ddpm_epoch_N.pt",
    )
    sample_p.add_argument(
        "--out",
        type=str,
        default="./samples/generated.png",
        help="Output PNG path",
    )
    sample_p.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="How many images to generate",
    )
    sample_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    sample_p.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable x0 clipping (usually hurts clarity)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
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
        if args.save_every is not None:
            cfg.save_every = args.save_every
        if args.sample_every is not None:
            cfg.sample_every = args.sample_every
        if args.sample_dir is not None:
            cfg.sample_dir = args.sample_dir
        if args.num_samples is not None:
            cfg.num_samples = args.num_samples

        trainer = Trainer(cfg)
        trainer.train()
        return

    if args.command == "sample":
        from src.diffusion.reverse import sample
        from src.diffusion.scheduler import NoiseScheduler
        from src.models.unet import build_unet
        from src.training.checkpoint import load_checkpoint
        from src.utils.config import TrainConfig
        from src.utils.device import get_device
        from src.utils.seed import set_seed
        from src.utils.visualize import save_sample_grid

        cfg = TrainConfig()
        if args.seed is not None:
            set_seed(args.seed)
        else:
            set_seed(cfg.seed)

        device = get_device()
        scheduler = NoiseScheduler.from_config(cfg, device=device)
        model = build_unet(cfg).to(device)
        load_checkpoint(args.checkpoint, model, map_location=device)
        model.eval()

        images = sample(
            model=model,
            scheduler=scheduler,
            shape=(args.num_samples, cfg.channels, cfg.image_size, cfg.image_size),
            device=device,
            clip_denoised=not args.no_clip,
            show_progress=True,
        )
        nrow = max(1, int(args.num_samples**0.5))
        out = save_sample_grid(images, args.out, nrow=nrow)
        print(f"Saved clarity samples → {out}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
