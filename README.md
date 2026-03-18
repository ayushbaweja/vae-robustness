# PGD Attacks on Generative Model VAEs

Projected Gradient Descent (PGD) adversarial attacks on the VAE encoders/decoders of various generative models. Measures how small, imperceptible pixel-space perturbations amplify through encoding into latent space and back through decoding.

## Supported VAEs

| Script | Model | Latent Channels | Input |
|---|---|---|---|
| `pgd_sd15_vae.py` | Stable Diffusion 1.5 | 4 | Image |
| `pgd_flux_vae.py` | FLUX.1 | 16 | Image |
| `pgd_flux2_vae.py` | FLUX.2 | 32 | Image |
| `pgd_cogvideox_vae.py` | CogVideoX | 16 | Image (as 1-frame video) |
| `pgd_ltx_vae.py` | LTX Video | 128 | Image (as 1-frame video) |

## Setup

```bash
uv sync
```

## Usage

Each script supports two attack modes:
- **pixel** — maximize reconstruction error through the full VAE (encode + decode)
- **latent** — maximize latent displacement through the encoder only

```bash
# SD 1.5
uv run python pgd_sd15_vae.py --epsilon 0.06 --loss pixel

# FLUX.1
uv run python pgd_flux_vae.py --epsilon 0.06 --loss pixel

# FLUX.2
uv run python pgd_flux2_vae.py --epsilon 0.06 --loss pixel

# CogVideoX
uv run python pgd_cogvideox_vae.py --epsilon 0.06 --loss pixel

# LTX Video
uv run python pgd_ltx_vae.py --epsilon 0.06 --loss pixel
```

### Common arguments

| Argument | Default | Description |
|---|---|---|
| `--input_dir` | `resources/test-images` | Directory of input images |
| `--output_dir` | `results/<model>_pgd` | Output directory |
| `--epsilon` | `0.06` | L-infinity perturbation budget (in [-1,1] scale) |
| `--alpha` | `0.01` | PGD step size |
| `--num_iter` | `40` | Number of PGD iterations |
| `--image_size` | varies | Resize input images to this size |
| `--loss` | `pixel` | Attack mode: `pixel` or `latent` |

## Output

Each run saves to `results/<model>_pgd/eps_<epsilon>_<loss>/`:
- `*_adv.png` — adversarial images
- `*_visualization.png` — 4-row comparison (original, latent channels, latent PCA, decoded)
- `summary.json` — per-image and average metrics (pixel MSE, latent MSE, L-inf norms, reconstruction errors)
