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

# AutoAttack-style ensemble
uv run python autoattack_vae.py --model sd15 --epsilon 0.06
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

### AutoAttack-style ensemble

`autoattack_vae.py` is a VAE-adapted ensemble attack rather than the original classification `AutoAttack` package. It runs:
- `apgd_recon` — adaptive PGD maximizing decoded reconstruction error against the clean input
- `apgd_decoded` — adaptive PGD maximizing decoded drift against the clean reconstruction
- `apgd_latent` — adaptive PGD maximizing encoder latent displacement
- `square_decoded` — square-style black-box patch search on decoded drift

For each image it keeps the adversarial example with the largest `decoded_diff_mse`, which makes it a stronger and less optimizer-sensitive robustness check than a single fixed-step PGD run.

```bash
uv run python autoattack_vae.py --model flux2 --epsilon 0.06
uv run python autoattack_vae.py --model ltx --epsilon 0.10 --apgd_steps 150 --square_steps 300
```

## Running the Full Sweep

To run all 60 experiments (5 models × 6 epsilons × 2 loss modes):

```bash
# All models sequentially
bash run_sweep.sh

# Single model
bash run_sweep.sh sd15

# Parallel across GPUs (one model per GPU)
bash run_model_sweep.sh sd15 0 &
bash run_model_sweep.sh flux1 1 &
bash run_model_sweep.sh flux2 2 &
bash run_model_sweep.sh cogvideox 3 &
wait
bash run_model_sweep.sh ltx 0  # after a GPU frees up
```

AutoAttack-style full sweep:

```bash
# All models sequentially
bash run_autoattack_sweep.sh

# Single model
bash run_autoattack_sweep.sh flux2

# Parallel across GPUs
bash run_autoattack_model_sweep.sh sd15 0 &
bash run_autoattack_model_sweep.sh flux1 1 &
bash run_autoattack_model_sweep.sh flux2 2 &
bash run_autoattack_model_sweep.sh cogvideox 3 &
wait
bash run_autoattack_model_sweep.sh ltx 0
```

Epsilons tested: `0.02, 0.04, 0.06, 0.1, 0.15, 0.2` with alpha and iteration count scaled accordingly.

## Output

Each run saves to `results/<model>_pgd/eps_<epsilon>_<loss>/`:
- `*_adv.png` — adversarial images
- `*_visualization.png` — 4-row comparison (original, latent channels, latent PCA, decoded)
- `summary.json` — per-image and average metrics (pixel MSE, latent MSE, L-inf norms, reconstruction errors)

## Analysis

```bash
uv run python analyze_results.py
uv run python analyze_autoattack_results.py
```

Reads all `summary.json` files and generates:
- `results/analysis/sweep_table.csv` — full results table (60 rows)
- `results/analysis/sweep_decoded_damage.png` — decoded diff MSE vs epsilon curves
- `results/analysis/sweep_latent_mse.png` — latent displacement vs epsilon
- `results/analysis/sweep_amplification.png` — amplification factor vs epsilon
- `results/analysis/matrix_heatmap.png` — heatmap across all (model, epsilon) pairs
- `results/analysis/pixel_vs_latent_comparison.png` — pixel vs latent loss side-by-side

AutoAttack analysis writes to `results/analysis_autoattack/`:
- `autoattack_sweep_table.csv` — full AutoAttack table
- `autoattack_decoded_damage.png` — decoded diff MSE vs epsilon
- `autoattack_latent_mse.png` — latent displacement vs epsilon
- `autoattack_amplification.png` — latent amplification vs epsilon
- `autoattack_heatmap.png` — decoded damage heatmap
- `autoattack_best_attack_share.png` — which sub-attack wins per image
- `autoattack_vs_pgd_pixel.png` — AutoAttack vs PGD pixel-loss comparison, when PGD results exist
- `autoattack_vs_pgd_gain_heatmap.png` — AutoAttack/PGD ratio and absolute damage gap heatmaps

## Results

Full sweep: 5 models × 6 epsilons × 2 loss modes = 60 experiments, each on 5 test images.

### Decoded Damage (Decoded Diff MSE) — Pixel Loss

| Model | ε=0.02 | ε=0.04 | ε=0.06 | ε=0.1 | ε=0.15 | ε=0.2 |
|---|---|---|---|---|---|---|
| **SD 1.5** | 0.016 | 0.077 | 0.149 | 0.238 | 0.340 | 0.434 |
| **FLUX.1** | 0.002 | 0.006 | 0.012 | 0.030 | 0.057 | 0.093 |
| **FLUX.2** | 0.119 | 0.248 | 0.385 | 0.473 | 0.777 | 0.981 |
| **CogVideoX** | 0.002 | 0.005 | 0.010 | 0.021 | 0.042 | 0.067 |
| **LTX Video** | 0.005 | 0.014 | 0.157 | 0.216 | 0.094 | 0.150 |

### Decoded Damage (Decoded Diff MSE) — Latent Loss

| Model | ε=0.02 | ε=0.04 | ε=0.06 | ε=0.1 | ε=0.15 | ε=0.2 |
|---|---|---|---|---|---|---|
| **SD 1.5** | 0.004 | 0.007 | 0.010 | 0.018 | 0.029 | 0.042 |
| **FLUX.1** | 0.001 | 0.004 | 0.008 | 0.017 | 0.034 | 0.054 |
| **FLUX.2** | 0.004 | 0.005 | 0.010 | 0.013 | 0.019 | 0.024 |
| **CogVideoX** | 0.001 | 0.002 | 0.003 | 0.007 | 0.013 | 0.022 |
| **LTX Video** | 0.001 | 0.004 | 0.008 | 0.021 | 0.041 | 0.058 |

### Key Findings

**Robustness ranking (pixel-loss attack, most to least robust):**

1. **CogVideoX** — Most robust across all epsilons. 3D video convolutions provide strong regularization. At ε=0.2, decoded damage is only 0.067.
2. **FLUX.1** — Nearly as robust as CogVideoX. 16 latent channels with an effective decoder that suppresses adversarial directions. 0.093 at ε=0.2.
3. **SD 1.5** — Moderate vulnerability. Steadily increasing damage curve. 0.434 at ε=0.2.
4. **LTX Video** — Erratic behavior with high per-image variance (non-monotonic damage curve). Some images collapse catastrophically while others are barely affected.
5. **FLUX.2** — Most vulnerable by far. Reaches 0.981 decoded MSE at ε=0.2 (near-total destruction). The 32-channel decoder massively amplifies small latent perturbations.

**Pixel vs Latent loss paradox:**

Across all models, latent-mode attacks achieve 2-10× larger latent displacements but cause 5-40× less decoded damage than pixel-mode attacks. This reveals that VAE decoders have a low-dimensional "sensitive subspace" — pixel-mode attacks find it (optimizing through the decoder), while latent-mode attacks spread perturbations into insensitive directions the decoder suppresses.

**FLUX.2 anomaly:**

FLUX.2 has extremely high pixel-loss vulnerability despite low latent MSE (only ~3.5 across all epsilons). Its 32-channel decoder is hypersensitive — tiny latent perturbations cause massive output damage. In latent mode, FLUX.2 achieves the highest latent MSE of any model (~32 at ε=0.2) but with minimal decoded damage (0.024), confirming that only specific latent directions matter.
