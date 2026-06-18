# FLUX.1-dev vs FLUX.2-dev VAE Robustness Report

## Executive Summary

This repository compares the robustness of image autoencoders by attacking the image input, then measuring how much the VAE reconstruction changes. For a VAE-only comparison, the important objects are not the full FLUX.1-dev and FLUX.2-dev generation models, but their autoencoders:

- **FLUX.1-dev VAE**: `AutoencoderKL`, 16 latent channels, 8x spatial downsampling.
- **FLUX.2-dev VAE**: `AutoencoderKLFlux2`, 32 latent channels, 8x spatial downsampling. (It also has a 2x2 latent packing + BatchNorm normalization path, but that is applied in the **generation pipeline**, not inside the VAE `encode`/`decode` that this study attacks — see "Correctness note" below.)

The main empirical finding is:

> FLUX.2's VAE has better clean reconstruction quality, but its decoded output changes much more under small adversarial pixel perturbations.

This points less to a generic "bad encoder" and more to an **encoder-decoder sensitivity problem**: FLUX.2's encoder maps tiny pixel perturbations into latent changes that the decoder treats as meaningful image signal. The decoder is therefore a major part of the observed failure mode, because the large visible damage appears when the perturbed latent is decoded.

The likely reason is a quality/robustness tradeoff. FLUX.2's VAE was retrained from scratch to improve the learnability-quality-compression tradeoff. That makes the latent space more expressive and useful for generation, but it does not imply adversarial stability.

## Scope

This report compares the **VAEs** used by FLUX.1-dev and FLUX.2-dev. It does not evaluate the full text-to-image/image-editing models, prompt following, safety behavior, or generation quality.

The relevant repository scripts are:

- `pgd_flux_vae.py`: FLUX.1 VAE PGD attack.
- `pgd_flux2_vae.py`: FLUX.2 VAE PGD attack.
- `autoattack_vae.py`: ensemble attack across APGD-style and square attacks.

The code now defaults to the intended dev model IDs:

- `black-forest-labs/FLUX.1-dev`
- `black-forest-labs/FLUX.2-dev`

Older saved FLUX.1 result artifacts often record `black-forest-labs/FLUX.1-schnell`. That is probably acceptable for a VAE-only comparison because public FLUX.1-dev and FLUX.1-schnell VAE configs match, but the cleanest final experiment should rerun FLUX.1 with the `FLUX.1-dev` repo ID and record model revisions.

## High-Level Model Architecture

### FLUX.1-dev

FLUX.1-dev is a 12B parameter rectified-flow transformer image model. Its public Hugging Face model card describes it as a rectified flow transformer trained for text-to-image generation and guidance-distilled for efficient inference.

For this repo, the full transformer is not loaded. Only the VAE is loaded:

```python
AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae")
```

The FLUX.1 VAE is structurally similar to the SD3/FLUX-style autoencoder:

- convolutional encoder-decoder backbone,
- ResNet blocks,
- mid-block attention,
- GroupNorm + SiLU,
- 8x spatial downsampling,
- 16 latent channels,
- `use_quant_conv = false`,
- `use_post_quant_conv = false`,
- scale/shift latent normalization in config.

For a 512x512 image, the latent has shape:

```text
(B, 16, 64, 64)
```

Before entering the FLUX.1 transformer, these latents are usually packed in 2x2 spatial groups:

```text
(B, 16, 64, 64) -> (B, 4096 / 4, 16 * 4) = (B, 1024, 64)
```

That packing is part of the full generation pipeline, not the VAE reconstruction experiment itself.

### FLUX.2-dev

FLUX.2-dev is a 32B parameter rectified-flow/flow-matching transformer image model. BFL describes it as combining a Mistral-3 24B vision-language model with a rectified-flow transformer, with native text-to-image, image editing, and multi-reference conditioning.

Again, this repo evaluates only the VAE:

```python
AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae")
```

The FLUX.2 VAE changes the latent representation:

- convolutional encoder-decoder backbone,
- ResNet blocks,
- mid-block attention,
- 8x spatial downsampling,
- 32 latent channels,
- `use_quant_conv = true`,
- `use_post_quant_conv = true`,
- a `BatchNorm2d` over `patch_size * latent_channels = 4 * 32 = 128` channels and a `patch_size = [2, 2]` packing step — **both used by the pipeline, not by `encode`/`decode`** (see "Correctness note").

For a 512x512 image, the VAE's `encode().latent_dist.mode()` returns the latent this study actually attacks:

```text
(B, 32, 64, 64)
```

In full generation (outside this VAE study), the FLUX.2 pipeline then packs 2x2 latent neighborhoods into the channel dimension after encoding:

```text
(B, 32, 64, 64) -> (B, 128, 32, 32)
```

The transformer then treats this packed representation as 128-wide image tokens:

```text
(B, 128, 32, 32) -> (B, 1024, 128)
```

This gives FLUX.2 twice the latent channel capacity of FLUX.1 and half the compression ratio:

```text
FLUX.1 compression = (3 * 8 * 8) / 16 = 12x
FLUX.2 compression = (3 * 8 * 8) / 32 = 6x
```

## Side-by-Side VAE Comparison

| Property | FLUX.1-dev VAE | FLUX.2-dev VAE |
|---|---:|---:|
| Diffusers class | `AutoencoderKL` | `AutoencoderKLFlux2` |
| Internal latent channels | 16 | 32 |
| Spatial downsample | 8x | 8x |
| VAE `encode/decode` latent for 512x512 (**what this study attacks**) | `16 x 64 x 64` | `32 x 64 x 64` |
| Pipeline-packed latent for 512x512 (generation only, *not attacked here*) | `16 x 64 x 64` | `128 x 32 x 32` |
| Packed token width (generation only) | 64 | 128 |
| Approx. compression | 12x | 6x |
| Quant conv | disabled | enabled |
| Post-quant conv | disabled | enabled |
| Latent normalization | scale/shift factors (config metadata; not applied in `encode`/`decode` round-trip) | BatchNorm + 2x2 packing, **applied in pipeline, not in `encode`/`decode`** |
| Public design goal | FLUX.1 latent space | retrained latent space for learnability, quality, compression |

## Attack Setup in This Repo

The main PGD attack optimizes an input perturbation `delta` under an L-infinity constraint:

```text
x_adv = clamp(x + delta)
||delta||_inf <= epsilon
```

Two objectives are used:

### Pixel/reconstruction objective

```text
maximize || D(E(x_adv)) - x ||^2
```

This attacks the full VAE path: image -> encoder -> latent -> decoder -> reconstruction.

### Latent objective

```text
maximize || E(x_adv) - E(x) ||^2
```

This attacks only the encoder's latent displacement. It is useful diagnostically, but less directly tied to visible output damage.

The most important cross-model metrics are:

- `recon_mse_orig`: clean reconstruction error.
- `decoded_diff_mse`: how much decoded output changes between clean and adversarial inputs.
- `recon_mse_adv_vs_orig`: how far adversarial reconstruction is from the original clean image.

Raw `latent_mse` should not be treated as directly comparable across FLUX.1 and FLUX.2 because the latent spaces have different dimensionality, scaling, and normalization.

## Repository Evidence

From `results/analysis/sweep_table.csv`, 30-image PGD sweep:

| Model | Epsilon | Loss | Clean Recon MSE | Decoded Diff MSE |
|---|---:|---|---:|---:|
| FLUX.1 | 0.02 | pixel | 0.002795 | 0.002936 |
| FLUX.2 | 0.02 | pixel | 0.001850 | 0.048860 |
| FLUX.1 | 0.06 | pixel | 0.002795 | 0.015757 |
| FLUX.2 | 0.06 | pixel | 0.001850 | 0.290685 |
| FLUX.1 | 0.20 | pixel | 0.002795 | 0.114244 |
| FLUX.2 | 0.20 | pixel | 0.001850 | 0.985769 |

The clean reconstruction metric favors FLUX.2:

```text
FLUX.1 recon_mse_orig ~= 0.002795
FLUX.2 recon_mse_orig ~= 0.001850
```

But decoded adversarial damage strongly favors FLUX.1:

```text
epsilon 0.02: FLUX.2 decoded_diff_mse is ~16.6x FLUX.1
epsilon 0.06: FLUX.2 decoded_diff_mse is ~18.4x FLUX.1
epsilon 0.20: FLUX.2 decoded_diff_mse is ~8.6x FLUX.1
```

The AutoAttack-style sweep shows the same direction:

| Model | Epsilon | Clean Recon MSE | Decoded Diff MSE |
|---|---:|---:|---:|
| FLUX.1 | 0.02 | 0.000804 | 0.003250 |
| FLUX.2 | 0.02 | 0.000539 | 0.343943 |
| FLUX.1 | 0.20 | 0.000804 | 0.158128 |
| FLUX.2 | 0.20 | 0.000539 | 1.429179 |

Again, FLUX.2 reconstructs clean inputs better, but adversarial perturbations create much larger decoded changes.

## Is the Decoder the Issue?

The decoder is very likely central to the failure mode, but the precise statement should be:

> The vulnerability is in the local behavior of the encoder-decoder composition. The decoder turns adversarially induced latent changes into large visible image changes.

Why this points toward the decoder:

1. The visible failure is measured after decoding.
2. FLUX.2's clean reconstruction is better, so the decoder is not simply lower quality.
3. FLUX.2's latent space is more expressive and less compressed. Small input changes can be preserved rather than smoothed out.
4. The pixel objective explicitly optimizes through the decoder, and FLUX.2 shows much larger decoded damage.

However, it is not correct to blame only the decoder. The attack starts in pixel space and needs the encoder Jacobian to map `delta` into latent movement. The decoder then determines whether that latent movement becomes visible corruption.

Mathematically, local sensitivity is governed by the Jacobian product:

```text
D(E(x + delta)) - D(E(x)) ~= J_D(E(x)) * J_E(x) * delta
```

So the issue could be:

- encoder sensitivity: `J_E(x)` is large in adversarial directions,
- decoder sensitivity: `J_D(z)` amplifies those directions,
- latent alignment: the perturbation moves along directions the decoder interprets as real image detail,
- or all of the above.

The current evidence most strongly supports **decoder amplification of adversarial latent directions**, not necessarily a defective decoder in ordinary reconstruction.

## Why FLUX.2 May Be Less Robust

### 1. More latent capacity means less smoothing (the dominant factor)

FLUX.1 compresses RGB images by about 12x (16 channels); FLUX.2 by about 6x (32 channels). The bottleneck acts as an implicit filter: to hit a 12x squeeze, FLUX.1 *must* discard information, and the cheapest things to discard are the small, high-frequency, low-amplitude details where an adversarial perturbation hides. FLUX.2's roomier latent has the capacity to keep that detail.

This single property explains both halves of the observed pattern at once: keeping more detail makes FLUX.2 reconstruct clean images *better*, and keeping more detail also lets the adversarial signal *survive* encoding. Robustness via compression is well established — the same reason JPEG compression or color-depth reduction can blunt adversarial attacks. FLUX.1's robustness is therefore best read as an accidental byproduct of its tighter bottleneck, not a deliberate defense.

A first-principles walkthrough of this argument (intuition → pictures → the `J_D · J_E` Jacobian math) lives in `understanding_vae_robustness.md`.

### 2. FLUX.2 was optimized for learnability, not adversarial stability

BFL states that FLUX.2 retrained the latent space from scratch to improve learnability and image quality while addressing the learnability-quality-compression tradeoff. That means the VAE is trained to produce representations that the flow model can learn from and decode with high fidelity.

Nothing in the public materials indicates adversarial training of the VAE against PGD-style L-infinity perturbations.

### 3. A high-fidelity decoder can amplify off-manifold latents

VAEs used for latent diffusion are not usually trained to make the decoder locally flat around every encoded image. They are trained to reconstruct natural images and support generation. If PGD pushes encoded latents into nearby but unnatural directions, a high-fidelity decoder may convert those directions into visible artifacts instead of ignoring them.

This is consistent with the observed pattern:

```text
better clean reconstruction + worse adversarial decoded damage
```

### 4. FLUX.2's quant-conv path differs (but BN/packing are NOT in the attacked path)

FLUX.2 uses `AutoencoderKLFlux2` with enabled quant/post-quant convs. These 1x1 convolutions are inside `encode`/`decode` and do alter local latent geometry, so they are part of the attacked mapping.

By contrast, FLUX.2's `BatchNorm2d` normalization and 2x2 patch packing are **not** invoked by `encode`/`decode`. In diffusers 0.37.1, `AutoencoderKLFlux2.encode()` returns a `DiagonalGaussianDistribution` over the raw 32-channel encoder output, and `decode()` consumes that same 32-channel latent; the BN (`self.vae.bn`) and `_pack_latents` are applied later, in `pipeline_flux2.py`. So they cannot explain the robustness gap measured here, and earlier phrasings that attributed sensitivity to a "BN/packing path" have been corrected. The drivers that the attack actually sees are **channel count / compression ratio** and the **encoder-decoder pair** (including the quant convs).

## Validity of This Repo's Results

### Correct VAE loading

The FLUX.2 path is correct:

```python
from diffusers import AutoencoderKLFlux2
AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae")
```

The FLUX.1 path should use:

```python
from diffusers import AutoencoderKL
AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae")
```

The repo has been updated to default to this path.

### Correctness note: the encode→decode round-trip is correctly paired

The comparison is apples-to-apples in the ways that matter:

- **Identical attack harness.** `pgd_flux_vae.py` and `pgd_flux2_vae.py` are identical except for the VAE class and model ID; the AutoAttack ensemble drives both models through one shared code path. Same epsilon, iterations, step size, loss, and metrics.
- **Clean, consistent round-trip for both.** For both VAEs the attack uses `encode().latent_dist.mode()` → `decode().sample`. For FLUX.2 this round-trips on the raw 32-channel latent; the BN/packing path is *not* in this path (see "Why FLUX.2 may be less robust" #4). No `scaling_factor`/`shift_factor` is applied on either side for either model, and because it is a round-trip, any constant latent scaling would cancel.
- **Empirical sanity check.** Clean reconstruction MSE is low and sensible for both (FLUX.1 ~0.0008, FLUX.2 ~0.0005 in the AutoAttack runs). A mishandled FLUX.2 latent (wrong normalization or packing mismatch) would produce garbage clean reconstructions; it does not. FLUX.2 in fact reconstructs *better*, which is what makes the robustness gap a real tradeoff rather than a loading artifact.
- **Fair primary metric.** `decoded_diff_mse` is measured in RGB pixel space under an identical pixel-space L-infinity budget, so it is comparable across the two latent spaces; raw `latent_mse` is not and should not be compared directly.

### Existing result caveat

Existing saved FLUX.1 results often record:

```text
black-forest-labs/FLUX.1-schnell
```

For a full text-to-image model comparison, that would be wrong. For a VAE-only comparison, it is probably acceptable if the FLUX.1-dev and FLUX.1-schnell VAE weights are identical. Public VAE configs match, but config equality does not prove weight equality.

Recommended cleanup:

1. Rerun all FLUX.1 sweeps with `black-forest-labs/FLUX.1-dev`.
2. Save Hugging Face revision hashes in each summary.
3. Save VAE config in each summary.
4. Optionally hash the VAE weight file or state dict.
5. Report `decoded_diff_mse` as the primary robustness metric, not raw latent MSE.

### Are the conclusions still likely valid?

Yes. The effect size is large enough that the qualitative conclusion is unlikely to be caused only by the FLUX.1-dev vs FLUX.1-schnell metadata issue. FLUX.2 shows order-of-magnitude larger decoded damage at small epsilons while also showing better clean reconstruction.

But for publication-quality claims, rerun with pinned model revisions.

## Suggested Follow-Up Experiments

### 1. Decoder-only latent perturbation test

Add random and adversarial latent-space perturbations directly to `z = E(x)`:

```text
z_adv = z + eta
D(z_adv) vs D(z)
```

Compare FLUX.1 and FLUX.2 for equal normalized latent perturbation sizes. This isolates decoder sensitivity from encoder sensitivity.

### 2. Encoder Jacobian norm estimate

Estimate local encoder sensitivity:

```text
||E(x + delta) - E(x)|| / ||delta||
```

Use random directions and PGD directions separately.

### 3. Decoder Jacobian norm estimate

Estimate:

```text
||D(z + eta) - D(z)|| / ||eta||
```

This directly tests whether FLUX.2's decoder has larger local amplification.

### 4. Frequency analysis of perturbations

Measure whether successful FLUX.2 perturbations are high-frequency or structured. If FLUX.1 filters high-frequency perturbations better, this would support the bottleneck/smoothing hypothesis.

### 5. Equalized latent metric

Normalize latent perturbations by per-channel standard deviation or use decoded metrics only. Avoid comparing unnormalized latent MSE across 16-channel and 32-channel VAEs.

## Conclusion

FLUX.2's VAE appears to be a higher-fidelity, more expressive autoencoder than FLUX.1's VAE. That is exactly what its architecture and training goals suggest: 32 channels, lower compression, a retrained latent space, and Flux2-specific normalization/packing.

The downside is adversarial sensitivity. Small pixel-space perturbations produce latent changes that FLUX.2's decoder converts into large visible reconstruction changes. The decoder is therefore a key part of the issue, but the vulnerability is best described as sensitivity of the full encoder-decoder mapping rather than a standalone decoder bug.

The current repo results are directionally valid, but old FLUX.1 artifacts should be rerun with `FLUX.1-dev` and pinned revisions before making formal claims.

## Sources

- Black Forest Labs, "FLUX.2: Frontier Visual Intelligence": https://bfl.ai/blog/flux-2
- Official FLUX.2 inference repository: https://github.com/black-forest-labs/flux2
- Hugging Face, `black-forest-labs/FLUX.1-dev`: https://huggingface.co/black-forest-labs/FLUX.1-dev
- Hugging Face, `black-forest-labs/FLUX.2-dev`: https://huggingface.co/black-forest-labs/FLUX.2-dev
- Hugging Face public VAE config examples for FLUX.1 and FLUX.2 variants, including `latent_channels`, quant-conv flags, and FLUX.2 `patch_size`.
