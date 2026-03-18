# PGD Attack on Stable Diffusion 1.5 VAE — Detailed Walkthrough

This document provides a complete, line-by-line explanation of `pgd_sd15_vae.py`. It assumes no prior knowledge of adversarial attacks, VAEs, or the specific math involved.

---

## Table of Contents

1. [Background Concepts](#1-background-concepts)
   - [What is a VAE?](#what-is-a-vae)
   - [What is an Adversarial Attack?](#what-is-an-adversarial-attack)
   - [What is PGD?](#what-is-pgd)
2. [Code Walkthrough](#2-code-walkthrough)
   - [Imports](#imports-lines-1-8)
   - [Image Loading and Conversion](#image-loading-and-conversion-lines-10-22)
   - [Latent Visualization Helpers](#latent-visualization-helpers-lines-24-46)
   - [PGD Attack Function](#pgd-attack-function-lines-51-95)
   - [Visualization Function](#visualization-function-lines-100-193)
   - [Main Entry Point](#main-entry-point-lines-198-283)
3. [Math Reference Summary](#3-math-reference-summary)

---

## 1. Background Concepts

### What is a VAE?

A **Variational Autoencoder (VAE)** is a neural network with two halves:

```
                    Encoder              Decoder
Input image x  ──────────────►  z  ──────────────►  x̂ (reconstruction)
(512×512×3)                  (64×64×4)              (512×512×3)
```

- The **encoder** compresses a high-resolution image into a small **latent representation** `z`. For Stable Diffusion 1.5, a 512×512×3 image becomes a 64×64×4 tensor — that's a 48× compression in spatial dimensions (512/64 = 8× each side) and the 3 color channels become 4 latent channels.
- The **decoder** reconstructs the image from `z`. A well-trained VAE produces `x̂ ≈ x`.

The encoder actually outputs two things: a **mean** (μ) and a **log-variance** (log σ²). Together they define a Gaussian distribution over latents. During normal use, you sample from this distribution. For our attack, we use the **mode** (just the mean μ), which is deterministic — this removes randomness so gradients are clean and the attack is reproducible.

### What is an Adversarial Attack?

An adversarial attack adds a small, carefully crafted perturbation δ to an input so that a model behaves badly, while the perturbation is (ideally) imperceptible to humans:

```
x_adv = x + δ       where ||δ|| ≤ ε
```

- `x` is the original clean image
- `δ` (delta) is the perturbation — a tensor the same size as the image
- `ε` (epsilon) is the **perturbation budget** — the maximum allowed change per pixel
- `x_adv` is the adversarial image

The constraint `||δ||_∞ ≤ ε` means no single pixel changes by more than ε. With images in the range [-1, 1], an ε = 0.06 means each pixel can shift by at most 0.06 out of a full range of 2.0, or about 3% — barely visible to the eye.

**Our goal:** find a δ that makes the VAE reconstruction of `x + δ` look as different from `x` as possible. The input looks almost identical, but the VAE output is corrupted.

### What is PGD?

**Projected Gradient Descent (PGD)** is an iterative algorithm for crafting adversarial perturbations. It is the "gold standard" first-order attack. Here is the intuition and math:

**Goal:** solve this optimization problem:

```
maximize  L(x + δ)       (make the loss as large as possible)
subject to  ||δ||_∞ ≤ ε  (stay within the perturbation budget)
            x + δ ∈ [-1, 1]  (stay within valid pixel range)
```

where `L` is a loss function measuring how badly the model behaves.

**Algorithm:** PGD solves this iteratively:

```
1. Initialize:  δ₀ ~ Uniform(-ε, ε)           (random start)
                δ₀ = clamp(x + δ₀, -1, 1) - x (ensure valid pixels)

2. For each iteration t = 0, 1, ..., T-1:
   a. Compute gradient:  g = ∇_δ L(x + δₜ)
   b. Take a step:       δₜ₊₁ = δₜ + α · sign(g)     ← "signed gradient ascent"
   c. Project onto L∞ ball:  δₜ₊₁ = clamp(δₜ₊₁, -ε, +ε)
   d. Project onto valid range: δₜ₊₁ = clamp(x + δₜ₊₁, -1, 1) - x

3. Return:  x_adv = x + δ_T
```

Key details:

- **Step 2a:** We compute how the loss changes with respect to δ using backpropagation. The gradient points in the direction that increases the loss fastest.
- **Step 2b:** We use `sign(g)` instead of `g` itself. This means we step by exactly `α` in each dimension, regardless of gradient magnitude. This is the same trick as in FGSM (Fast Gradient Sign Method), but applied repeatedly. α is the **step size** per iteration.
- **Steps 2c–2d:** These are the "projections." After each step, we may have violated our constraints (δ too large, or pixel out of range), so we clamp back. This is what makes it *projected* gradient descent.
- **Why maximize?** We want the model to fail, so we maximize the loss (damage). Normal gradient descent minimizes; we do gradient *ascent* (note the `+` sign in step 2b).

---

## 2. Code Walkthrough

### Imports (Lines 1–8)

```python
import torch                          # PyTorch — tensor operations, autograd
import torch.nn.functional as F       # Loss functions like mse_loss
import numpy as np                    # Numpy — array operations for image I/O
import matplotlib.pyplot as plt       # Plotting library for visualization grids
from PIL import Image                 # Pillow — image loading and saving
from pathlib import Path              # Filesystem path handling
from diffusers import AutoencoderKL   # HuggingFace's SD1.5 VAE implementation
from sklearn.decomposition import PCA # Principal Component Analysis for latent viz
```

### Image Loading and Conversion (Lines 10–22)

#### `load_image` (Lines 10–14)

```python
def load_image(path: str, size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
```

**Line 12:** Opens an image file and converts to RGB (3 channels). Resizes to `size × size` (default 512×512), which is the native resolution for SD1.5's VAE.

**Line 13 — Normalization math:**

Raw pixels are integers in [0, 255]. We convert to floats in [-1, 1]:

```
pixel_normalized = pixel_raw / 127.5 - 1.0
```

| Raw value | Normalized value |
|-----------|-----------------|
| 0         | -1.0            |
| 127.5     | 0.0             |
| 255       | 1.0             |

This [-1, 1] range is what the SD1.5 VAE was trained on.

**Line 14 — Shape transformation:**

```
np.array(img) gives shape:  (H, W, 3)     — height, width, channels (RGB)
.permute(2, 0, 1) gives:    (3, H, W)     — channels first (PyTorch convention)
.unsqueeze(0) gives:         (1, 3, H, W)  — add batch dimension
```

The batch dimension is needed because PyTorch models expect batched inputs. A single image is a "batch of 1."

#### `tensor_to_numpy_image` (Lines 17–21)

```python
def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    return (t.permute(1, 2, 0).cpu().float() * 0.5 + 0.5).clamp(0, 1).numpy()
```

This is the reverse of `load_image`. It converts a model tensor back to a displayable image.

**Line 19–20:** If the tensor has a batch dimension `(1, 3, H, W)`, remove it to get `(3, H, W)`.

**Line 21 — Denormalization math:**

```
pixel_display = pixel_normalized * 0.5 + 0.5
```

| Normalized | Display |
|-----------|---------|
| -1.0      | 0.0     |
| 0.0       | 0.5     |
| 1.0       | 1.0     |

This maps [-1, 1] → [0, 1], the range matplotlib and PIL expect for floats. `.clamp(0, 1)` handles any values that went slightly out of range. `.permute(1, 2, 0)` goes from `(3, H, W)` back to `(H, W, 3)`.

### Latent Visualization Helpers (Lines 24–46)

The VAE latent `z` has shape `(1, 4, 64, 64)` — 4 channels, not 3. We can't directly display it as an RGB image. These two functions offer different strategies to convert 4 channels → 3 (RGB).

#### `latent_to_rgb_channels` (Lines 24–32)

```python
def latent_to_rgb_channels(z: torch.Tensor) -> np.ndarray:
    z3 = z[0, :3].cpu().float()  # (3, h, w)
    for c in range(3):
        mn, mx = z3[c].min(), z3[c].max()
        z3[c] = (z3[c] - mn) / (mx - mn + 1e-8)
    return z3.permute(1, 2, 0).numpy()
```

**Strategy:** Simply take the first 3 of 4 latent channels and treat them as R, G, B. This discards channel 3, but is simple and fast.

**Line 27:** `z[0, :3]` selects batch element 0, channels 0–2. Result shape: `(3, 64, 64)`.

**Lines 29–31 — Per-channel min-max normalization:**

For each channel independently:

```
z_normalized[c] = (z[c] - min(z[c])) / (max(z[c]) - min(z[c]))
```

This stretches each channel to fill the full [0, 1] range. The `+ 1e-8` prevents division by zero if a channel is constant.

**Why per-channel?** Latent channels can have very different value ranges (e.g., channel 0 might be in [-5, 5] while channel 1 is in [-20, 20]). Without per-channel normalization, one channel could dominate the visualization.

#### `latent_to_rgb_pca` (Lines 35–46)

```python
def latent_to_rgb_pca(z: torch.Tensor) -> np.ndarray:
    C, h, w = z.shape[1], z.shape[2], z.shape[3]
    pixels = z[0].cpu().float().reshape(C, -1).T.numpy()  # (h*w, C)
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(pixels)  # (h*w, 3)
    for c in range(3):
        mn, mx = rgb[:, c].min(), rgb[:, c].max()
        rgb[:, c] = (rgb[:, c] - mn) / (mx - mn + 1e-8)
    return rgb.reshape(h, w, 3)
```

**Strategy:** Use Principal Component Analysis to find the 3 directions of greatest variance in the 4D latent space, then project onto those.

**Line 39 — Reshape for PCA:**

```
z[0] has shape (4, 64, 64)
.reshape(4, -1) → (4, 4096)     — flatten spatial dims; -1 means "infer this dim"
.T → (4096, 4)                   — transpose so each row is a "pixel" with 4 features
```

Each of the 4096 spatial positions becomes a data point with 4 features (the 4 latent channels).

**Lines 40–41 — PCA:**

PCA finds the 3 orthogonal directions in 4D space that capture the most variance. `fit_transform` returns the data projected onto these 3 axes. Result: `(4096, 3)`.

**Math of PCA (briefly):** Given data matrix X (4096×4), PCA:
1. Centers the data: X̄ = X - mean(X)
2. Computes the covariance matrix: C = X̄ᵀX̄ / n
3. Finds eigenvectors of C, sorted by eigenvalue (largest = most variance)
4. Projects onto the top 3 eigenvectors

**Lines 43–45:** Same min-max normalization as before.

**Line 46:** Reshape from `(4096, 3)` back to `(64, 64, 3)` for display.

**Why PCA over raw channels?** PCA captures the most informative view of the 4D space. It doesn't arbitrarily discard channel 3, and it decorrelates the channels, often producing cleaner visualizations.

### PGD Attack Function (Lines 51–95)

This is the core of the script. It implements the PGD algorithm described in [Section 1](#what-is-pgd).

#### Function Signature (Lines 51–58)

```python
def pgd_attack_vae(
    vae: AutoencoderKL,       # The VAE model (encoder + decoder)
    images: torch.Tensor,     # Clean input image, shape (1, 3, 512, 512), range [-1, 1]
    epsilon: float = 0.06,    # L∞ perturbation budget
    alpha: float = 0.01,      # Step size per iteration
    num_iter: int = 40,       # Number of PGD iterations
    loss_mode: str = "pixel", # Which loss to use: "pixel" or "latent"
) -> torch.Tensor:            # Returns adversarial image, same shape as input
```

**Parameters explained:**

- **epsilon (ε = 0.06):** Each pixel can change by at most 0.06 in [-1, 1] range. In [0, 255] scale, that's 0.06 × 127.5 ≈ 7.65, so roughly 8/255. This is a common perturbation budget in adversarial ML.
- **alpha (α = 0.01):** Each iteration moves each pixel by at most 0.01. With 40 iterations, the maximum total displacement is 40 × 0.01 = 0.4, but the L∞ clamp at ε = 0.06 prevents it from exceeding 0.06. The ratio ε/α = 6 means it takes at least 6 steps to reach the boundary, giving the attack room to explore.
- **num_iter (T = 40):** More iterations = stronger attack (closer to optimal δ), but slower.

#### Two Loss Modes (Lines 62–63)

```
"pixel"  — max ||decode(encode(x+δ)) - x||²  (gradient through full VAE)
"latent" — max ||encode(x+δ) - encode(x)||²   (gradient through encoder only)
```

**Pixel mode** measures damage in the final decoded image — "how bad does the output look?"
**Latent mode** measures displacement in the compressed representation — "how far did the latent move?"

These give very different results because the decoder can dampen certain latent directions (see discussion below in the attack loop).

#### Setup (Lines 65–73)

```python
    vae.eval()
    device = images.device
```

**Line 65:** Sets the VAE to evaluation mode. This disables dropout and uses running statistics for batch normalization — important for consistent, deterministic behavior.

```python
    with torch.no_grad():
        z_orig = vae.encode(images).latent_dist.mode()
```

**Lines 68–69:** Pre-compute the latent representation of the clean image. `torch.no_grad()` means we don't track gradients here (saves memory, and we don't need gradients for the clean encoding).

- `vae.encode(images)` runs the encoder, returning a distribution object
- `.latent_dist` is a `DiagonalGaussianDistribution` with mean μ and log-variance log σ²
- `.mode()` returns the mean μ — the most likely latent under the distribution

This `z_orig` is used as the reference target in latent mode. In pixel mode it's computed but not used in the loss (it's still useful for visualization later).

```python
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, -1, 1) - images
    delta.requires_grad = True
```

**Line 71 — Random initialization:**

Creates a tensor of the same shape as `images` (1, 3, 512, 512), filled with random values uniformly distributed in [-ε, +ε].

**Why random start?** PGD with a random start is more likely to find strong adversarial examples than starting from δ = 0. Starting from zero, the attack might get stuck in a local maximum. Random restarts explore different regions of the perturbation space.

**Line 72 — Double projection:**

After adding the random δ to x, some pixels might go outside [-1, 1]. This line clamps the perturbed image to [-1, 1], then subtracts x back out to get the adjusted δ. This ensures:
1. `||δ||_∞ ≤ ε` (from the uniform initialization)
2. `x + δ ∈ [-1, 1]` (from the clamp)

**Line 73:** Tells PyTorch to track gradients with respect to `delta`. This is essential — during backpropagation, we need ∂L/∂δ.

#### Attack Loop (Lines 75–93)

```python
    for i in range(num_iter):
        adv = images + delta
```

**Line 76:** Construct the adversarial image. Note that `images` is fixed throughout — we only modify `delta`.

```python
        posterior = vae.encode(adv).latent_dist
        z = posterior.mode()
```

**Lines 78–79:** Encode the adversarial image through the VAE encoder. `.mode()` gives the deterministic mean of the latent distribution. Using the mode (rather than sampling) is important: sampling introduces randomness that would make gradients noisy and the attack weaker.

```python
        if loss_mode == "latent":
            loss = F.mse_loss(z, z_orig, reduction="sum")
        else:
            recon = vae.decode(z).sample
            loss = F.mse_loss(recon, images, reduction="sum")
```

**Lines 81–86 — Loss computation:**

**Latent mode (line 83):**

```
L_latent = Σᵢ (z_adv[i] - z_orig[i])²
```

This is the sum of squared differences between every element of the adversarial latent and the clean latent. `reduction="sum"` means we sum over all elements (rather than averaging). Summing gives larger gradient magnitudes, but since we use `sign()` anyway, it doesn't affect the direction — only the scale, which `α` controls.

**Pixel mode (lines 85–86):**

```
L_pixel = Σᵢ (x̂_adv[i] - x[i])²
```

where `x̂_adv = decode(z)` is the decoded adversarial image. This measures how different the VAE's reconstruction of the adversarial input is from the original clean image.

**Why the two modes behave differently:**

In pixel mode, the gradient ∂L/∂δ flows through: δ → encoder → z → decoder → x̂ → loss. The decoder's Jacobian (∂x̂/∂z) acts as a filter. If the decoder is insensitive to certain directions in latent space (as trained VAE decoders tend to be for out-of-distribution latents), those gradient components get dampened. The attack cannot exploit those directions.

In latent mode, the gradient ∂L/∂δ flows through: δ → encoder → z → loss. The decoder is bypassed entirely. The attack can push the latent in any direction that the encoder allows, even directions the decoder is insensitive to. This produces larger latent displacements but paradoxically may cause less visual damage in the decoded image.

```python
        grad = torch.autograd.grad(loss, delta)[0]
```

**Line 88 — Gradient computation:**

`torch.autograd.grad(loss, delta)` computes ∂L/∂δ via backpropagation. This returns a tuple; `[0]` extracts the gradient tensor (same shape as delta: 1×3×512×512).

Each element `grad[0, c, h, w]` tells us: "if I increase pixel (c, h, w) of δ by a tiny amount, how much does the loss increase?" Positive gradient = increasing that pixel increases the loss (which we want).

We use `torch.autograd.grad` instead of `loss.backward()` because δ is not a model parameter — it's a separate tensor we're optimizing. `autograd.grad` is the clean way to get gradients for non-parameter tensors.

```python
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(images + delta, -1, 1) - images
            delta.requires_grad = True
```

**Lines 89–93 — Update step (inside `torch.no_grad()` because this is manual optimization, not part of the computational graph):**

**Line 90 — Signed gradient ascent:**

```
δₜ₊₁ = δₜ + α · sign(∇_δ L)
```

`grad.sign()` replaces each gradient value with -1, 0, or +1. We then step by exactly `α` in each dimension. This is **gradient ascent** (note the `+` sign) because we want to **maximize** the loss.

**Why `sign()` instead of the raw gradient?** This is the L∞ analog of steepest ascent. In L∞ geometry, the steepest ascent direction is `sign(∇L)` — it's the direction that increases L the most per unit of L∞ norm. Using the raw gradient would be steepest ascent in L2 geometry, which would "waste" perturbation budget by concentrating changes on a few high-gradient pixels.

**Line 91 — Project onto L∞ ball:**

```
δₜ₊₁ = clamp(δₜ₊₁, -ε, +ε)
```

If any element of δ exceeds ε (or is below -ε), clamp it back. This enforces the perturbation budget.

**Line 92 — Project onto valid pixel range:**

```
δₜ₊₁ = clamp(x + δₜ₊₁, -1, 1) - x
```

Some pixels in x might be close to -1 or +1. Adding δ could push them out of range. This clamps the *perturbed image* to [-1, 1], then extracts back the adjusted δ.

**Example:** if `x[pixel] = 0.98` and `δ[pixel] = 0.05`, then `x + δ = 1.03`, clamped to 1.0, so `δ` becomes `1.0 - 0.98 = 0.02`.

**Line 93:** Reassign `requires_grad = True` because lines 90–92 created new tensors (detached from the graph). The next iteration needs gradients w.r.t. this new δ.

#### Return (Line 95)

```python
    return (images + delta).detach()
```

Returns the final adversarial image `x_adv = x + δ_T`. `.detach()` removes it from the computational graph (no longer need gradients).

### Visualization Function (Lines 100–193)

This function creates a 4-row × 3-column figure comparing clean vs. adversarial at every stage of the VAE pipeline.

#### Decorator (Line 100)

```python
@torch.no_grad()
```

Disables gradient tracking for the entire function. Visualization is inference-only — no gradients needed, so this saves memory and compute.

#### Encode and Decode (Lines 112–117)

```python
    z_orig = vae.encode(original).latent_dist.mode()
    z_adv = vae.encode(adversarial).latent_dist.mode()
    dec_orig = vae.decode(z_orig).sample
    dec_adv = vae.decode(z_adv).sample
```

Independently encode both images, then decode both latents. This gives us four tensors to compare:
- `original` vs `adversarial` (pixel space input)
- `z_orig` vs `z_adv` (latent space)
- `dec_orig` vs `dec_adv` (pixel space output)

#### Difference Maps (Lines 120–140)

```python
    perturbation = np.clip(np.abs(img_adv - img_orig) * 10, 0, 1)
```

**Line 122:** Pixel perturbation map. `|x_adv - x|` is very small (max ε = 0.06 in [-1,1] → 0.03 in [0,1]), so we multiply by 10 to make it visible. Without amplification, the perturbation would look like a solid black image.

The same ×10 amplification is applied to latent diffs (line 128) and decoded diffs (line 140).

#### Metrics (Lines 142–149)

```python
    pixel_mse = F.mse_loss(adversarial, original).item()
    pixel_linf = (adversarial - original).abs().max().item()
    latent_mse = F.mse_loss(z_adv, z_orig).item()
    latent_linf = (z_adv - z_orig).abs().max().item()
    recon_mse_orig = F.mse_loss(dec_orig, original).item()
    recon_mse_adv = F.mse_loss(dec_adv, original).item()
    dec_diff_mse = F.mse_loss(dec_adv, dec_orig).item()
```

Seven metrics are computed:

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| `pixel_mse` | (1/N) Σ (x_adv - x)² | Average squared pixel perturbation |
| `pixel_linf` | max |x_adv - x| | Worst-case pixel change (should ≤ ε) |
| `latent_mse` | (1/N) Σ (z_adv - z_orig)² | Average squared latent displacement |
| `latent_linf` | max |z_adv - z_orig| | Worst-case latent element change |
| `recon_mse_orig` | (1/N) Σ (dec_orig - x)² | VAE reconstruction error on clean input (baseline quality) |
| `recon_mse_adv` | (1/N) Σ (dec_adv - x)² | How far decoded adversarial is from clean original |
| `dec_diff_mse` | (1/N) Σ (dec_adv - dec_orig)² | How much the decoded output changed |

`recon_mse_orig` serves as a baseline: even without an attack, the VAE doesn't reconstruct perfectly. If `dec_diff_mse >> recon_mse_orig`, the attack is causing damage beyond normal VAE reconstruction error.

#### Plotting (Lines 151–183)

```python
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
```

Creates a 4×3 grid of subplots. The grid layout:

```
┌─────────────────┬─────────────────┬─────────────────┐
│ Original        │ Adversarial     │ Perturbation ×10│
├─────────────────┼─────────────────┼─────────────────┤
│ Latent ch0-2    │ Latent ch0-2    │ Latent Ch Diff  │
│ (clean)         │ (adversarial)   │ ×10             │
├─────────────────┼─────────────────┼─────────────────┤
│ Latent PCA      │ Latent PCA      │ Latent PCA Diff │
│ (clean)         │ (adversarial)   │ ×10             │
├─────────────────┼─────────────────┼─────────────────┤
│ Decoded         │ Decoded         │ Decoded Diff    │
│ (clean)         │ (adversarial)   │ ×10             │
└─────────────────┴─────────────────┴─────────────────┘
```

**Row 1** shows the attack is nearly invisible in pixel space.
**Rows 2–3** show the latent representation is dramatically different.
**Row 4** shows the impact on the VAE's decoded output.

### Main Entry Point (Lines 198–283)

#### Argument Parsing (Lines 201–212)

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | `resources/test-images` | Directory containing input images |
| `--output_dir` | `results/sd15_pgd` | Base output directory |
| `--model_id` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | HuggingFace model identifier |
| `--epsilon` | 0.06 | L∞ perturbation budget |
| `--alpha` | 0.01 | PGD step size |
| `--num_iter` | 40 | Number of PGD iterations |
| `--image_size` | 512 | Resize images to this resolution |
| `--loss` | `pixel` | Loss mode: `pixel` or `latent` |

#### Model Loading (Lines 217–221)

```python
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    vae = vae.to(device).eval()
```

Downloads (or loads from cache) the VAE component of Stable Diffusion 1.5 from HuggingFace. The `subfolder="vae"` means we only load the VAE, not the full diffusion model (UNet, text encoder, etc.). `.to(device)` moves weights to GPU. `.eval()` sets evaluation mode.

#### Processing Loop (Lines 237–265)

For each image:

1. **Load** the image as a tensor (line 239)
2. **Attack** it with PGD (lines 242–246)
3. **Save** the adversarial image as PNG (lines 249–252)
4. **Visualize** with the full comparison grid + compute metrics (lines 255–261)
5. **Print** key metrics (lines 263–265)

#### Summary (Lines 268–277)

```python
    summary = {
        "config": vars(args),
        "per_image": all_metrics,
        "average": { ... },
    }
```

Saves a JSON file with the full experiment configuration, per-image metrics, and averages across all images. This makes results reproducible and easy to compare across different hyperparameter settings.

---

## 3. Math Reference Summary

### Notation

| Symbol | Meaning |
|--------|---------|
| x | Clean input image, tensor in [-1, 1] |
| δ | Adversarial perturbation, same shape as x |
| x_adv = x + δ | Adversarial image |
| ε | Maximum per-pixel perturbation (L∞ budget) |
| α | PGD step size per iteration |
| T | Number of PGD iterations |
| E(·) | VAE encoder (maps image → latent distribution) |
| D(·) | VAE decoder (maps latent → image) |
| z = E(x).mode() | Latent mean (deterministic encoding) |
| x̂ = D(z) | Decoded/reconstructed image |

### Optimization Problem

```
maximize    L(δ)
subject to  ||δ||_∞ ≤ ε
            x + δ ∈ [-1, 1]
```

where the loss L depends on the mode:

**Pixel mode:**
```
L_pixel(δ) = Σᵢ ( D(E(x + δ)) - x )ᵢ²
```

**Latent mode:**
```
L_latent(δ) = Σᵢ ( E(x + δ) - E(x) )ᵢ²
```

### PGD Update Rule

```
δₜ₊₁ = Π_S ( δₜ + α · sign(∇_δ L(δₜ)) )
```

where Π_S is the projection onto the feasible set S = { δ : ||δ||_∞ ≤ ε, x + δ ∈ [-1, 1] }, implemented as two consecutive clamp operations.

### Why sign(∇) Is Optimal for L∞

In L∞-constrained optimization, the steepest ascent direction is:

```
d* = argmax_{||d||_∞ ≤ 1} ⟨∇L, d⟩ = sign(∇L)
```

This can be verified: the inner product ⟨∇L, d⟩ is maximized when each dᵢ has the same sign as (∇L)ᵢ and magnitude 1, which is exactly sign(∇L).
