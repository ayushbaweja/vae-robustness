# Stable Diffusion VAE vs. FLUX VAE: Architecture & Key Differences

## Table of Contents

1. [Background: Why a VAE?](#background-why-a-vae)
2. [Stable Diffusion VAE Architecture (SD 1.x / 2.x / SDXL)](#stable-diffusion-vae-architecture)
3. [FLUX VAE Architecture (FLUX.1 / FLUX.2)](#flux-vae-architecture)
4. [Side-by-Side Comparison](#side-by-side-comparison)
5. [Latent Space Differences in Depth](#latent-space-differences-in-depth)
6. [Reconstruction Quality](#reconstruction-quality)
7. [Practical Implications](#practical-implications)
8. [Sources](#sources)

---

## Background: Why a VAE?

Latent diffusion models (LDMs) do not operate directly in pixel space. Instead, a pretrained autoencoder compresses images into a lower-dimensional **latent space** where the diffusion process (forward noising + reverse denoising) takes place. After denoising, the decoder maps latents back to pixels. This compression is what makes models like Stable Diffusion and FLUX tractable on consumer hardware — the diffusion U-Net or Transformer operates on tensors that are orders of magnitude smaller than the raw image.

The autoencoder used across the Stable Diffusion family is an **AutoencoderKL** — a convolutional encoder-decoder with KL-divergence regularization toward a standard normal distribution on the learned latent. It is *not* a classical VAE in the strict sense; the KL weight is very mild and the decoder is trained with a combination of reconstruction losses and an adversarial (GAN) objective.

---

## Stable Diffusion VAE Architecture

### Applies to: SD 1.x, SD 2.x, SDXL

The original Stable Diffusion VAE descends from the **KL-f8** autoencoder introduced in the [Latent Diffusion Models paper (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752).

### Configuration

| Parameter | Value |
|---|---|
| **Latent channels** | 4 |
| **Spatial downsampling factor** | 8× |
| **Base channels (`ch`)** | 128 |
| **Channel multipliers (`ch_mult`)** | [1, 2, 4, 4] |
| **Block out channels** | [128, 256, 512, 512] |
| **Num ResNet blocks per level** | 2 |
| **Attention resolutions** | Midblock only (lowest spatial resolution) |
| **Scaling factor** | 0.18215 (SD 1.x/2.x), 0.13025 (SDXL) |
| **Shift factor** | None |

### Encoder

The encoder converts a `(B, 3, H, W)` RGB image into a `(B, 8, H/8, W/8)` tensor (8 = 2 × `z_channels`, representing mean and log-variance), then samples to produce `(B, 4, H/8, W/8)` latents.

**Layer-by-layer structure:**

1. **Initial convolution:** `Conv2d(3 → 128, kernel=3, padding=1)`
2. **Resolution level 1 (128 ch):** 2× ResNet blocks → Downsample (stride-2 Conv2d)
3. **Resolution level 2 (256 ch):** 2× ResNet blocks → Downsample (stride-2 Conv2d)
4. **Resolution level 3 (512 ch):** 2× ResNet blocks → Downsample (stride-2 Conv2d)
5. **Resolution level 4 (512 ch):** 2× ResNet blocks (no downsample — final spatial level)
6. **Mid-block:** ResNet block → Self-Attention → ResNet block
7. **Output head:** GroupNorm → SiLU → `Conv2d(512 → 8, kernel=3)` → `quant_conv(8 → 8, kernel=1)`

### Decoder

The decoder mirrors the encoder in reverse, converting `(B, 4, H/8, W/8)` latents back to `(B, 3, H, W)` images.

**Layer-by-layer structure:**

1. **Input head:** `post_quant_conv(4 → 4, kernel=1)` → `Conv2d(4 → 512, kernel=3)`
2. **Mid-block:** ResNet block → Self-Attention → ResNet block
3. **Resolution level 4 (512 ch):** 3× ResNet blocks → Upsample (nearest-neighbor 2× + Conv2d)
4. **Resolution level 3 (512 ch):** 3× ResNet blocks → Upsample
5. **Resolution level 2 (256 ch):** 3× ResNet blocks → Upsample
6. **Resolution level 1 (128 ch):** 3× ResNet blocks (no upsample — final)
7. **Output head:** GroupNorm → SiLU → `Conv2d(128 → 3, kernel=3)`

> **Note:** The decoder uses `n_resnet_blocks + 1` = 3 residual blocks per level, one more than the encoder.

### Building Blocks

**ResNet Block:**
```
GroupNorm(32) → SiLU → Conv2d(3×3) → GroupNorm(32) → SiLU → Conv2d(3×3) + skip
```
If input and output channels differ, a 1×1 convolution is applied to the skip connection.

**Self-Attention (Midblock only):**
```
GroupNorm → Q/K/V projections → Scaled dot-product attention → Output projection + skip
```
Scaling: `channels ** -0.5`. Operates on flattened spatial dimensions.

**Downsampling:** `Conv2d(kernel=3, stride=2, padding=0)` with asymmetric zero-padding.

**Upsampling:** `nn.Upsample(scale_factor=2, mode='nearest')` followed by `Conv2d(kernel=3, padding=1)`.

**Normalization:** GroupNorm with 32 groups and ε = 1e-6 throughout.

**Activation:** SiLU (Swish): `x * σ(x)`.

### Latent Sampling

The encoder outputs mean (μ) and log-variance (log σ²). Log-variance is clamped to [-30, 20]. Sampling uses the reparameterization trick:

```
z = μ + σ · ε,  where ε ~ N(0, I)
```

In practice, the KL regularization is so mild that variances are extremely small — taking just the mean vs. sampling produces nearly identical results.

### Training Losses

- **Reconstruction:** L1 or MSE pixel loss + perceptual loss (LPIPS)
- **Adversarial:** PatchGAN discriminator
- **KL divergence:** Very low weight (mild regularization toward N(0, I))

### SDXL VAE Specifics

The SDXL VAE uses the **same architecture** as the SD 1.x VAE but was **retrained from scratch**. Two versions were released (v0.9 and v1.0; v0.9 is generally considered better). Key differences from the SD 1.x VAE:
- Different scaling factor (0.13025 vs. 0.18215)
- No "bright spot" artifact that plagued the original SD VAE at higher resolutions
- Latents are **incompatible** with SD 1.x/2.x latents despite the same architecture

---

## FLUX VAE Architecture

### Applies to: FLUX.1 (shares config with SD3 VAE), FLUX.2

The FLUX VAE was trained from scratch by Black Forest Labs using an adversarial objective, scaling the latent representation from 4 to **16 channels**. FLUX.1's VAE shares the same configuration as the SD3 VAE but reportedly achieves higher decoding quality.

### Configuration (FLUX.1 / SD3)

| Parameter | Value |
|---|---|
| **Latent channels** | 16 |
| **Spatial downsampling factor** | 8× |
| **Base channels (`ch`)** | 128 |
| **Channel multipliers (`ch_mult`)** | [1, 2, 4, 4] |
| **Block out channels** | [128, 256, 512, 512] |
| **Num ResNet blocks per level** | 2 |
| **Attention resolutions** | Midblock only |
| **Scaling factor** | 0.3611 (FLUX.1), 1.5305 (SD3) |
| **Shift factor** | 0.1159 (FLUX.1), 0.0609 (SD3) |

### Key Architectural Differences from SD VAE

1. **16 latent channels** instead of 4 — the single most impactful change
2. **Shift factor** added alongside scaling factor for latent normalization:
   - Encode: `latent = (raw_latent - shift_factor) * scaling_factor`
   - Decode: `raw_latent = (latent / scaling_factor) + shift_factor`
3. **Removed `quant_conv` and `post_quant_conv`** layers — the 1×1 convolutions that previously bridged encoder/decoder to the latent space are gone
4. The convolutional backbone (ResNet blocks, attention, up/downsampling) remains structurally the same

### Encoder (FLUX.1)

Same convolutional backbone as the SD VAE, but the final output convolution produces `2 × 16 = 32` channels (mean + log-variance for 16 latent channels) instead of `2 × 4 = 8`:

- Input: `(B, 3, H, W)`
- Output: `(B, 16, H/8, W/8)` after sampling

### Decoder (FLUX.1)

Mirrors the encoder, taking `(B, 16, H/8, W/8)` → `(B, 3, H, W)`. The input convolution maps from 16 channels (instead of 4) to 512.

### Latent Packing for the Transformer

A distinctive feature of the FLUX pipeline (not the VAE itself, but tightly coupled to it) is **2×2 patch packing** of latents before they enter the diffusion transformer:

```
(B, 16, H/8, W/8) → (B, (H/16)*(W/16), 16*4) = (B, seq_len, 64)
```

This groups 2×2 spatial patches and flattens them into the channel dimension, reducing the sequence length by 4× while preserving all information. The transformer's input projection then maps these 64-dimensional tokens to its internal hidden dimension (3072).

### FLUX.2 VAE (Next Generation)

FLUX.2 introduces a further-improved VAE with notable changes:

| Parameter | FLUX.1 | FLUX.2 |
|---|---|---|
| **Latent channels** | 16 | 32 |
| **Compression ratio** | 12× | 6× |
| **Training scheme** | Standard | RePA-like (representation alignment) |
| **Normalization** | External scaling/shift factors | Encoded in checkpoint |

FLUX.2's 32-channel VAE achieves the **least aggressive compression** of any major diffusion model VAE, preserving maximal detail at the cost of larger latent tensors. It was re-trained from scratch to optimize the "learnability-quality-compression" trilemma.

---

## Side-by-Side Comparison

| Property | SD 1.x/2.x | SDXL | FLUX.1 (& SD3) | FLUX.2 |
|---|---|---|---|---|
| **Latent channels** | 4 | 4 | 16 | 32 |
| **Spatial downsampling** | 8× | 8× | 8× | 8× |
| **Compression ratio** | 48× | 48× | 12× | 6× |
| **Latent shape** (512×512 input) | 4×64×64 | 4×64×64 | 16×64×64 | 32×64×64 |
| **Scaling factor** | 0.18215 | 0.13025 | 0.3611 | — |
| **Shift factor** | — | — | 0.1159 | — |
| **quant_conv / post_quant_conv** | Yes | Yes | No | No |
| **Architecture backbone** | ConvNet + midblock attn | ConvNet + midblock attn | ConvNet + midblock attn | ConvNet + midblock attn |
| **Training objective** | Recon + KL + GAN | Recon + KL + GAN | Adversarial | RePA-like adversarial |
| **Latent compatibility** | SD 1.x ↔ 2.x only | SDXL only | SD3 ↔ FLUX.1 | FLUX.2 only |

### Compression Ratio Calculation

The compression ratio measures how much smaller the latent representation is compared to the original RGB pixel count:

```
Compression = (3 × f²) / C

SD:    (3 × 64) / 4  = 48×
FLUX.1: (3 × 64) / 16 = 12×
FLUX.2: (3 × 64) / 32 =  6×
```

Where `f = 8` (downsampling factor) and `C` = number of latent channels.

---

## Latent Space Differences in Depth

### Why More Channels Matter

The SD VAE's 4-channel latent space must encode all visual information — color, texture, edges, lighting, fine detail — into just 4 feature maps per spatial location. This aggressive bottleneck forces lossy compression, and the decoder must **hallucinate** fine details that were lost during encoding.

FLUX's 16-channel (or 32-channel) latent space provides significantly more capacity per spatial location:
- **Richer texture encoding:** Fine-grained patterns (hair, fabric weave, skin pores) can be preserved rather than approximated
- **Better color fidelity:** More channels allow subtler color gradients and tonal information
- **Sharper edges:** Less information loss means the decoder needs to reconstruct less
- **Reduced artifacts:** The decoder has more to work with, producing fewer blurring artifacts

### Scaling and Shift Factors

**SD 1.x/2.x/SDXL** use only a scaling factor to normalize latents to roughly unit variance before the diffusion process operates on them.

**SD3/FLUX.1** add a **shift factor** that accounts for non-zero mean in the latent distribution. This two-parameter normalization (`z_normalized = (z - shift) * scale`) produces latents that are better centered and scaled for the diffusion process. The removal of `quant_conv`/`post_quant_conv` simplifies the pipeline — the latent normalization is handled purely by these scalar factors rather than learned 1×1 convolutions.

### KL Regularization Behavior

Across all versions, the KL regularization is extremely mild. The learned variances are so small that the latent space is effectively deterministic — sampling from the posterior vs. taking the mean produces negligible differences. The KL term primarily serves to prevent the latent space from collapsing or developing pathological distributions, not to impose a strong information bottleneck.

---

## Reconstruction Quality

| VAE | rFID ↓ | PSNR ↑ | LPIPS ↓ |
|---|---|---|---|
| SD 1.5 | 0.3131 | 26.43 dB | 0.0328 |
| SDXL | 0.3511 | 26.76 dB | 0.0320 |
| SD3 (16-ch) | 0.0257 | 30.32 dB | 0.0132 |
| FLUX.1 (16-ch) | — | ≥ SD3 | ≤ SD3 |

Key observations:
- The jump from 4-channel to 16-channel VAEs produces a **dramatic improvement** in reconstruction fidelity (~4 dB PSNR gain, ~2.5× better LPIPS)
- FLUX.1's VAE uses the same config as SD3 but achieves qualitatively higher decoding quality
- The 4-channel VAEs (SD 1.5 and SDXL) perform similarly despite SDXL being retrained — the bottleneck is the 4-channel capacity, not training
- rFID drops by an order of magnitude with 16-channel VAEs, indicating much more faithful reconstructions

### Known Artifacts

- **SD 1.x VAE:** "Bright spot" artifact that worsens at higher resolutions. Caused by attention layer instabilities in the midblock.
- **SDXL VAE:** No bright spot artifact. Version 0.9 generally considered better than 1.0.
- **FLUX.1 VAE:** Large activation values observed in midblock attention (the attention can be disabled with minimal quality loss, similar to SD VAE).
- **FLUX.2 VAE:** "Uniquely bad scale-equivariance" has been noted — the VAE's behavior changes unexpectedly with input resolution. Investigation ongoing.

---

## Practical Implications

### Memory and Compute

| VAE | Latent size (1024×1024 input) | Relative memory |
|---|---|---|
| SD/SDXL (4-ch) | 4 × 128 × 128 = 65,536 | 1× |
| FLUX.1 (16-ch) | 16 × 128 × 128 = 262,144 | 4× |
| FLUX.2 (32-ch) | 32 × 128 × 128 = 524,288 | 8× |

The larger latent representations mean:
- The diffusion transformer processes 4–8× more values per spatial position
- FLUX mitigates this through 2×2 patch packing (reducing sequence length by 4×)
- Overall, FLUX trades VAE compression efficiency for reconstruction quality, placing more burden on the diffusion model

### Cross-Compatibility

VAE latents are **not interchangeable** between model families:
- SD 1.x ↔ SD 2.x: Compatible (same VAE weights in practice)
- SDXL: Incompatible with SD 1.x/2.x (retrained VAE)
- SD3 ↔ FLUX.1: Share the same VAE config and are approximately compatible
- FLUX.2: Incompatible with all prior models (32-channel latents)

### Fine-Tuning Considerations

- When fine-tuning diffusion models, the VAE is typically **frozen** — only the diffusion backbone (U-Net or Transformer) is trained
- Swapping VAEs within the same channel family (e.g., SD 1.x ft-EMA vs. ft-MSE) can alter output aesthetics without retraining
- Moving between 4-channel and 16-channel VAEs requires retraining the diffusion model from scratch
- Recent work (REPA-E, EQ-VAE) explores jointly fine-tuning the VAE and diffusion model for improved results

---

## Sources

- [Notes on SD VAE — madebyollin (comprehensive technical notes)](https://gist.github.com/madebyollin/ff6aeadf27b2edbc51d05d5f97a595d9)
- [Autoencoder for Stable Diffusion — LabML (annotated implementation)](https://nn.labml.ai/diffusion/stable_diffusion/model/autoencoder.html)
- [VAE Encoder and Decoder — DeepWiki](https://deepwiki.com/hkproj/pytorch-stable-diffusion/4.3-vae-encoder-and-decoder)
- [Demystifying Flux Architecture — arXiv](https://arxiv.org/html/2507.09595v1)
- [Diffusers welcomes Stable Diffusion 3 — Hugging Face Blog](https://huggingface.co/blog/sd3)
- [FLUX.2: Analyzing and Enhancing the Latent Space of FLUX — Black Forest Labs](https://bfl.ai/research/representation-comparison)
- [VAE: The Latent Bottleneck — Medium (Efrat Taig)](https://medium.com/@efrat_taig/vae-the-latent-bottleneck-why-image-generation-processes-lose-fine-details-a056dcd6015e)
- [VAE Matters: Latent Compression Choices for DiT — Stanford CS231N](https://cs231n.stanford.edu/2025/papers/text_file_840591073-CS_231N_Project_Report.pdf)
- [EQ-VAE: Equivariance Regularized Latent Space — arXiv](https://arxiv.org/html/2502.09509v2)
- [REPA-E: Unlocking VAE for End-to-End Tuning — arXiv](https://arxiv.org/html/2504.10483v1)
- [SD3 & FLUX: Complete Guide to MMDiT Architecture — SOTAAZ Blog](https://blog.sotaaz.com/post/sd3-flux-architecture-en)
- [Stable Diffusion VAE — Built In](https://builtin.com/artificial-intelligence/stable-diffusion-vae)
- [AutoencoderKL — Hugging Face Diffusers](https://huggingface.co/docs/diffusers/en/api/models/autoencoderkl)
