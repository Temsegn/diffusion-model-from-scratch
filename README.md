# DDPM From Scratch

Educational Denoising Diffusion Probabilistic Model (DDPM) trained on CIFAR-10 with PyTorch.

## Clarity (better images)

In diffusion, **clarity** means how sharp and recognizable generated samples look.

- **Early epochs** → blurry / noisy grids  
- **Later epochs** → clearer CIFAR-like images  

This project improves sample clarity by:

1. **Reverse sampling** with predicted-`x0` **clipping** to `[-1, 1]` (avoids color blow-up)
2. Saving **clarity grids** at epochs **10 / 50 / 100** under `samples/`

## Train (full run, checkpoint every epoch)

```bash
python main.py train
```

Useful flags:

```bash
python main.py train --epochs 100 --save-every 1 --sample-every 10
```

Sample PNGs land in `./samples/samples_epoch_XXX.png`.

## Generate images from a checkpoint

```bash
python main.py sample --checkpoint checkpoints/ddpm_epoch_100.pt --out samples/final.png
```

## Layout

| Path | Role |
|------|------|
| `src/diffusion/scheduler.py` | β / α / ᾱ (+ reverse variances) |
| `src/diffusion/forward.py` | Add noise (training) |
| `src/diffusion/reverse.py` | Denoise from noise → clear image |
| `src/diffusion/losses.py` | Noise-prediction MSE |
| `src/training/trainer.py` | Train loop + clarity sample grids |
| `samples/` | Generated image grids |
| `checkpoints/` | `ddpm_epoch_N.pt` |
