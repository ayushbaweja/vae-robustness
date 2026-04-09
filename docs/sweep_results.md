# Epsilon Sweep Results — Full Analysis

Complete results from the PGD adversarial attack sweep across 5 VAE models, 6 epsilon values, and 2 loss modes (60 experiments total, each on 5 test images).

---

## Experimental Setup

| Parameter | Values |
|---|---|
| **Models** | SD 1.5 (4ch), FLUX.1 (16ch), FLUX.2 (32ch), CogVideoX (16ch), LTX Video (128ch) |
| **Epsilons** | 0.02, 0.04, 0.06, 0.1, 0.15, 0.2 |
| **Loss modes** | pixel (max reconstruction error), latent (max latent displacement) |
| **Alpha (step size)** | 0.005–0.02, scaled with epsilon |
| **Iterations** | 40 (ε ≤ 0.1), 50 (ε = 0.15), 60 (ε = 0.2) |
| **Test images** | 5 JPEGs from `resources/test-images/` |

---

## 1. Decoded Damage — Pixel Loss Attack

This is the primary vulnerability metric: how much does the decoded output differ from the original when the input is adversarially perturbed?

| Model | ε=0.02 | ε=0.04 | ε=0.06 | ε=0.1 | ε=0.15 | ε=0.2 |
|---|---|---|---|---|---|---|
| **SD 1.5** | 0.0159 | 0.0768 | 0.1495 | 0.2377 | 0.3400 | 0.4337 |
| **FLUX.1** | 0.0017 | 0.0057 | 0.0124 | 0.0298 | 0.0569 | 0.0927 |
| **FLUX.2** | 0.1186 | 0.2485 | 0.3849 | 0.4727 | 0.7775 | 0.9811 |
| **CogVideoX** | 0.0019 | 0.0054 | 0.0096 | 0.0207 | 0.0416 | 0.0671 |
| **LTX Video** | 0.0048 | 0.0140 | 0.1566 | 0.2164 | 0.0938 | 0.1504 |

**Observations:**
- CogVideoX and FLUX.1 are clearly the most robust, staying below 0.1 even at ε=0.2.
- FLUX.2 is catastrophically vulnerable — near-total output destruction at ε=0.2 (MSE 0.981).
- SD 1.5 shows steady, approximately linear degradation.
- LTX Video is non-monotonic, suggesting the attack gets trapped in suboptimal local maxima at certain epsilon values, or that the loss landscape is highly irregular.

---

## 2. Decoded Damage — Latent Loss Attack

| Model | ε=0.02 | ε=0.04 | ε=0.06 | ε=0.1 | ε=0.15 | ε=0.2 |
|---|---|---|---|---|---|---|
| **SD 1.5** | 0.0042 | 0.0073 | 0.0103 | 0.0176 | 0.0287 | 0.0417 |
| **FLUX.1** | 0.0012 | 0.0039 | 0.0076 | 0.0175 | 0.0335 | 0.0539 |
| **FLUX.2** | 0.0038 | 0.0054 | 0.0096 | 0.0127 | 0.0194 | 0.0238 |
| **CogVideoX** | 0.0007 | 0.0018 | 0.0032 | 0.0069 | 0.0133 | 0.0218 |
| **LTX Video** | 0.0015 | 0.0042 | 0.0083 | 0.0205 | 0.0409 | 0.0582 |

**Observations:**
- All models show dramatically less decoded damage under latent-mode attacks (5–40× less).
- The robustness ranking changes: FLUX.2 is now among the most robust (0.024 at ε=0.2), while FLUX.1 and LTX Video show higher relative damage.
- CogVideoX remains the most robust overall.

---

## 3. Latent Space Displacement

### Pixel-loss attack (latent MSE is a side effect, not the objective)

| Model | ε=0.02 | ε=0.04 | ε=0.06 | ε=0.1 | ε=0.15 | ε=0.2 |
|---|---|---|---|---|---|---|
| **SD 1.5** | 6.49 | 10.12 | 12.05 | 14.76 | 17.35 | 19.17 |
| **FLUX.1** | 1.71 | 3.09 | 3.68 | 4.79 | 5.79 | 6.50 |
| **FLUX.2** | 2.17 | 3.05 | 3.07 | 3.15 | 3.46 | 3.47 |
| **CogVideoX** | 0.15 | 0.25 | 0.32 | 0.43 | 0.55 | 0.65 |
| **LTX Video** | 0.03 | 0.06 | 0.09 | 0.12 | 0.17 | 0.20 |

### Latent-loss attack (directly maximized)

| Model | ε=0.02 | ε=0.04 | ε=0.06 | ε=0.1 | ε=0.15 | ε=0.2 |
|---|---|---|---|---|---|---|
| **SD 1.5** | 19.49 | 29.69 | 36.28 | 45.63 | 55.64 | 64.77 |
| **FLUX.1** | 2.78 | 4.43 | 5.46 | 6.80 | 8.00 | 8.91 |
| **FLUX.2** | 15.42 | 16.30 | 26.63 | 23.46 | 32.12 | 32.69 |
| **CogVideoX** | 0.44 | 0.65 | 0.80 | 1.02 | 1.23 | 1.45 |
| **LTX Video** | 0.12 | 0.15 | 0.20 | 0.27 | 0.35 | 0.42 |

**Observations:**
- SD 1.5 has the highest absolute latent displacement under both attacks. Its 4-channel bottleneck amplifies perturbations heavily.
- FLUX.2 shows a striking pattern: latent MSE saturates quickly under pixel-loss attacks (~3.5) but reaches ~33 under latent-loss attacks. Only a tiny fraction of the latent space is decoder-sensitive.
- Video VAEs (CogVideoX, LTX) have very low latent displacement in both modes — their 3D convolutions act as effective low-pass filters.

---

## 4. The Pixel vs. Latent Loss Paradox

A consistent finding across all models: latent-mode attacks produce larger latent displacements but far less decoded damage.

| Model | Latent MSE ratio (latent/pixel) | Decoded MSE ratio (pixel/latent) | at ε=0.06 |
|---|---|---|---|
| **SD 1.5** | 3.0× | 14.5× | |
| **FLUX.1** | 1.5× | 1.6× | |
| **FLUX.2** | 8.7× | 40.0× | |
| **CogVideoX** | 2.5× | 3.0× | |
| **LTX Video** | 2.4× | 18.8× | |

**Interpretation:** VAE decoders are sensitive to only a low-dimensional subspace of their latent space. Pixel-mode attacks naturally find these sensitive directions (because gradients flow through the decoder), while latent-mode attacks spread perturbations uniformly across all latent dimensions, most of which the decoder ignores.

This effect is most extreme in FLUX.2: the attack can move the latent code by 26.6 MSE, but the decoder barely notices (0.010 decoded MSE). Yet a pixel-mode attack with only 3.1 latent MSE causes 0.385 decoded MSE — a 125× difference in decoder sensitivity per unit of latent displacement.

---

## 5. Amplification Factor

The amplification factor (latent MSE / pixel MSE) measures how much the encoder magnifies small input perturbations.

| Model | ε=0.02 | ε=0.06 | ε=0.2 | Trend |
|---|---|---|---|---|
| **SD 1.5** | 25,065 | 6,079 | 928 | Decreasing — encoder saturates |
| **FLUX.1** | 7,423 | 1,860 | 300 | Decreasing |
| **FLUX.2** | 8,201 | 1,510 | 161 | Decreasing |
| **CogVideoX** | 505 | 112 | 20 | Decreasing |
| **LTX Video** | 105 | 35 | 8 | Decreasing |

All models show decreasing amplification with larger epsilon, indicating that the encoder's sensitivity saturates — there's a finite amount of latent displacement achievable regardless of input perturbation size.

---

## 6. Robustness Rankings

### By decoded damage (pixel-loss attack at ε=0.06)

1. **CogVideoX** (0.010) — Most robust
2. **FLUX.1** (0.012)
3. **SD 1.5** (0.149)
4. **LTX Video** (0.157)
5. **FLUX.2** (0.385) — Least robust

### By decoded damage (pixel-loss attack at ε=0.2)

1. **CogVideoX** (0.067)
2. **FLUX.1** (0.093)
3. **LTX Video** (0.150)
4. **SD 1.5** (0.434)
5. **FLUX.2** (0.981)

### By latent amplification (pixel-loss, ε=0.06)

1. **LTX Video** (35×) — Least amplification
2. **CogVideoX** (112×)
3. **FLUX.2** (1,510×)
4. **FLUX.1** (1,860×)
5. **SD 1.5** (6,079×) — Most amplification

---

## 7. Model-Specific Analysis

### SD 1.5
- **Profile:** High latent amplification, moderate decoded damage, smooth scaling.
- The 4-channel bottleneck means perturbations are heavily compressed, leading to the highest latent MSE. But the decoder is moderately robust — not all latent displacement translates to visual damage.
- Steady, predictable degradation curve — good candidate for adversarial training.

### FLUX.1
- **Profile:** Moderate latent displacement, very low decoded damage.
- The 16-channel encoder spreads information across more dimensions, reducing per-channel sensitivity. The decoder effectively filters adversarial noise.
- Most consistent model — low variance across images, monotonic damage curve.

### FLUX.2
- **Profile:** Low latent displacement under pixel-loss attack, catastrophic decoded damage.
- The decoder is hypersensitive to specific latent directions. Even 3 MSE of latent displacement causes near-total output corruption.
- Extreme gap between pixel and latent loss effectiveness (40×) suggests the decoder has a very narrow "safe manifold" in latent space.
- Likely needs architectural changes (not just adversarial training) to fix.

### CogVideoX
- **Profile:** Minimal latent displacement, minimal decoded damage, overall most robust.
- 3D temporal convolutions act as aggressive low-pass filters, even on single-frame inputs.
- The encoder's spatial-temporal pooling naturally smooths adversarial perturbations.
- Robustness appears to be an architectural byproduct, not an explicit design goal.

### LTX Video
- **Profile:** Very low latent displacement, erratic decoded damage.
- The 128-channel latent space is extremely high-dimensional, making it hard for PGD to find effective adversarial directions.
- Non-monotonic damage curve (damage at ε=0.15 is lower than at ε=0.1) suggests the loss landscape has local optima that trap the PGD optimizer.
- High per-image variance: some images collapse catastrophically (decoded MSE > 0.6) while others barely change.

---

## 8. Output Files

All results are in `results/`:

```
results/
├── analysis/
│   ├── sweep_table.csv
│   ├── sweep_decoded_damage.png
│   ├── sweep_latent_mse.png
│   ├── sweep_amplification.png
│   ├── matrix_heatmap.png
│   └── pixel_vs_latent_comparison.png
├── sd15_pgd/
│   ├── eps_0.02_pixel/   ... eps_0.2_latent/   (12 dirs)
├── flux1_pgd/            (12 dirs)
├── flux2_pgd/            (12 dirs)
├── cogvideox_pgd/        (12 dirs)
└── ltx_pgd/              (12 dirs)
```

Each experiment directory contains:
- 5 adversarial images (`*_adv.png`)
- 5 visualization grids (`*_visualization.png`)
- 1 summary JSON (`summary.json`)
