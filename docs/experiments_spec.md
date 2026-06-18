# Experiment Spec: Isolating Encoder vs Decoder Sensitivity (FLUX.1 vs FLUX.2 VAE)

> **Audience:** the Claude Code agent running on the GPU machine.
> **Goal:** turn the *inferred* Jacobian story from `understanding_vae_robustness.md`
> into *measured* results. We currently only measure the full round-trip
> (`decoded_diff_mse ≈ ‖J_D · J_E · δ‖`). These three experiments separate the two
> factors and test the "bottleneck-as-sieve" hypothesis directly.
>
> Read `docs/understanding_vae_robustness.md` and `docs/flux1_vs_flux2_vae_report.md`
> first — they define the framing, the metrics, and the correctness caveats. This
> file assumes that context.

## Background you must respect

- The two VAEs are loaded exactly as in the existing scripts:
  - FLUX.1: `from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae")`
  - FLUX.2: `from diffusers import AutoencoderKLFlux2; AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae")`
  - Both are gated repos; the machine needs an HF token with access. Only the
    `vae/` subfolder downloads (~hundreds of MB each), not the full models.
- Attack/round-trip convention (keep identical to `pgd_flux_vae.py`):
  - Images loaded to `[-1, 1]`, shape `(1, 3, 512, 512)`, 512 divisible by 16.
  - Latent = `vae.encode(x).latent_dist.mode()` (deterministic mean).
  - Reconstruction = `vae.decode(z).sample`.
  - **Do not** apply `scaling_factor`/`shift_factor` or the FLUX.2 pipeline
    BN/packing — they are not in the attacked path (see report §"Why FLUX.2... #4").
- **Fairness rules (critical — the whole point is a fair 16-ch vs 32-ch comparison):**
  1. Compare **output/pixel-space** quantities directly (RGB, same units for both).
  2. When measuring or injecting **latent** perturbations, normalize **per
     channel by that channel's std** computed over the clean latents, so a
     "unit" perturbation means the same thing in both latent spaces. Never
     compare raw latent L2/Linf across the two models.
  3. Hold the input budget identical across models (same `epsilon`, same
     normalized latent perturbation size).
- Device: use `cuda`. Cast VAEs to `.float()` (fp32) for stable gradients/Jacobians.
- Test images: `resources/test-images` (5 images). Optionally also point
  `--input_dir` at the ImageNet-25 set if it's present on the machine, for n=25.
- Save every run's config + per-image + averaged metrics to JSON under
  `results/jacobian/...`, mirroring the existing `summary.json` style, and emit
  plots under the same dir.

---

## Experiment 1 — Decoder sensitivity (isolate `J_D`)

**Question:** Given equal *normalized* latent movement, does FLUX.2's decoder
turn it into more pixel-space change than FLUX.1's? (Tests the "high-fidelity
decoder amplifies" claim.)

**Method (per image):**
1. `z = E(x)`; `dec0 = D(z)`.
2. Compute per-channel latent std `s_c` over the spatial dims of `z` (shape
   `(C,)`). This is the normalization scale.
3. For a sweep of relative magnitudes `r ∈ {0.05, 0.1, 0.2, 0.4}` (fraction of
   per-channel std):
   - **Random directions** (`N_rand = 16` samples): draw `η ~ N(0, 1)` shaped
     like `z`, scale each channel by `r * s_c`, so `η[:,c] *= r * s_c`. Measure
     pixel-space response and the gain:
     ```
     resp = ‖D(z + η) − dec0‖_2            # pixel L2 over (3,H,W)
     gain = resp / ‖η_norm‖_2               # η_norm = η / s_c (per-channel), so denom is in std-units
     ```
     Report mean/std of `resp`, `gain`, and `decoded_mse = MSE(D(z+η), dec0)`
     over the N samples.
   - **Worst-case direction** (power iteration, ~10 steps): find the unit latent
     direction (in std-normalized space) that maximizes `‖D(z+η) − dec0‖`. Use
     autograd: start from random `η`, repeatedly set
     `η ← normalize(grad_η ‖D(z+η)−dec0‖²)` scaled to `r * s_c`. Report the
     top gain. This estimates the **decoder's largest local gain** ≈ top singular
     value of `J_D` in std-normalized coordinates.

**Primary output:** a table/plot of decoder gain vs `r`, FLUX.1 vs FLUX.2, for
both random and worst-case directions. **Hypothesis:** FLUX.2 gain ≥ FLUX.1 gain,
especially worst-case.

**Watch for:** if random-direction gains are similar but worst-case diverges,
that says the decoders are similar on average but FLUX.2 has sharper worst-case
directions — still consistent with the story; report it precisely.

---

## Experiment 2 — Encoder sensitivity (isolate `J_E`)

**Question:** Does FLUX.1's tighter bottleneck shrink adversarial pixel
perturbations more than FLUX.2's? (Tests the "sieve blocks δ at the encoder"
claim — the factor we argue is dominant.)

**Method (per image):** measure normalized latent response to a unit pixel nudge,
for three direction types, at fixed input budget (use the same `epsilon` grid as
the repo, e.g. `0.02, 0.06, 0.2`, or fix `‖δ‖` and report gain):
1. `z = E(x)`; per-channel std `s_c` as in Exp 1.
2. Define **normalized latent movement**:
   `Δz_norm = ‖ (E(x+δ) − E(x)) / s_c ‖_2`  (per-channel divide, then L2).
3. Directions:
   - **Random:** `δ` uniform in `[-ε, ε]`, clamp `x+δ` to `[-1,1]`. `N_rand=16`.
   - **PGD-latent:** run the existing `loss_mode="latent"` PGD (maximize
     `‖E(x+δ)−E(x)‖²`) to get the worst-case encoder direction within the L∞ ball.
   - **PGD-pixel:** the full-round-trip adversarial δ (`loss_mode="pixel"`), to
     see how much latent movement the *actually damaging* attack induces.
4. Report encoder gain `= Δz_norm / ‖δ‖_2` for each direction type, per model.

**Primary output:** table/plot of encoder gain (std-normalized) vs direction
type and `ε`, FLUX.1 vs FLUX.2. **Hypothesis:** for adversarial (PGD) directions,
FLUX.2's normalized encoder gain ≥ FLUX.1's — i.e. the same pixel nudge moves the
FLUX.2 latent further in std-units. (Note the repo already hints at this:
FLUX.2's `latent_linf` under attack is larger, but that's *unnormalized*; this
experiment makes it a fair per-std comparison.)

**Combine with Exp 1:** `decoded_diff_mse` should be roughly explained by
`(encoder gain) × (decoder gain)`. Sanity-check that the product direction
matches the measured round-trip gap (don't expect exact equality — it's a local
linearization — but the ordering should hold).

---

## Experiment 3 — Frequency analysis of perturbations

**Question:** Are the perturbations that successfully attack each VAE
high-frequency, and does FLUX.1 filter high frequencies that FLUX.2 passes?
(Tests the "sieve = low-pass filter" mechanism.)

**Part A — spectrum of adversarial δ (reuse existing or regenerate):**
1. For each model, take the PGD adversarial images (regenerate with
   `pgd_*_vae.py` at matched ε, or load `results/*/eps_0.06_pixel/*_adv.png` and
   subtract the clean input to recover δ — recovering from saved 8-bit PNGs
   loses precision, so **prefer regenerating** δ in-memory).
2. Compute the **2D FFT** of δ per channel, average magnitude over channels, then
   compute the **radially-averaged power spectrum** (power vs spatial frequency).
3. Plot FLUX.1 vs FLUX.2 radial spectra (log power vs frequency), averaged over
   images. **Hypothesis:** FLUX.2's effective δ carries relatively more
   high-frequency power; FLUX.1's effective δ is pushed toward lower frequencies
   (because high-freq gets filtered, so the attack can't use it).

**Part B — band-survival test (the cleaner, more direct probe):**
1. Build band-limited unit perturbations: for a set of radial frequency bands
   (e.g. 8 log-spaced rings in 2D frequency space), synthesize `δ_band` whose FFT
   energy lies only in that ring, normalized to fixed pixel `‖δ_band‖`.
2. Measure how much each band **survives the round-trip**:
   ```
   survival(band) = ‖ D(E(x + δ_band)) − D(E(x)) ‖ / ‖ δ_band ‖
   ```
   averaged over images, per model.
3. Plot survival vs frequency band, FLUX.1 vs FLUX.2. **Hypothesis:** FLUX.1's
   curve drops faster at high frequency (stronger low-pass / sieve), FLUX.2 stays
   higher — i.e. FLUX.2 passes high-freq perturbations the bottleneck of FLUX.1
   would suppress. This is the most direct test of the sieve picture.

---

## Deliverables

1. `experiments/decoder_sensitivity.py`, `experiments/encoder_sensitivity.py`,
   `experiments/frequency_analysis.py` (or one `jacobian_experiments.py` with
   `--exp {decoder,encoder,freq}`), reusing the load/transform helpers from the
   existing scripts (factor shared helpers into a small `vae_common.py` if clean).
2. JSON summaries + plots under `results/jacobian/<exp>/<model>_...`.
3. A short results writeup `docs/jacobian_experiments_results.md`:
   - the three plots,
   - a table of encoder gain, decoder gain (random + worst-case), and their
     product vs the measured `decoded_diff_mse`,
   - one paragraph per experiment stating whether the hypothesis held,
   - update `flux1_vs_flux2_vae_report.md` §"Suggested Follow-Up Experiments" to
     reference the results (turn predictions into findings).

## Gotchas

- Worst-case/power-iteration and PGD need gradients → keep VAEs in fp32, call
  `vae.eval()`, and use `torch.autograd.grad` (don't accumulate graph across
  images — detach between images to avoid OOM).
- FLUX.2 latent is 32-ch `(1,32,64,64)`; FLUX.1 is 16-ch `(1,16,64,64)`. All
  per-channel-std code must read `C` from the tensor, not hardcode.
- Always report pixel-space output metrics for cross-model claims; use
  std-normalized latent units only for the encoder-gain numbers, and say so.
- Keep ε, image set, iteration counts identical across the two models in any
  given comparison; record them in the JSON config.
- Memory: FLUX.2 decoder is heavier; if OOM, drop batch to 1 image (already is)
  and reduce `N_rand`, or run models sequentially (load FLUX.1, run, free,
  load FLUX.2).
