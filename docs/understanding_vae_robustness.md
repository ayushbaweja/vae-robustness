# Understanding VAE Adversarial Robustness From First Principles

> A slow build-up: plain intuition first, pictures second, math last. By the end
> you should understand *why* FLUX.1's autoencoder resists these attacks better
> than FLUX.2's, and be able to defend that claim with both a picture and an
> equation.
>
> Companion to `flux1_vs_flux2_vae_report.md` (the results) — this file is the
> "why it works that way" explainer.

---

## Level 0 — The one-sentence version

> FLUX.1 squeezes the image through a *narrower* pipe. The narrow pipe throws
> away the tiny adversarial scribble before it can reach the decoder. FLUX.2's
> wider pipe keeps more detail — great for quality, bad for robustness.

Everything below is just unpacking that sentence.

---

## Level 1 — What a VAE actually does

A VAE (variational autoencoder) is the part of an image model that converts
between **pixels** and a compact **latent** code. It has two halves:

```
        ENCODER                          DECODER
   ┌───────────────┐                ┌───────────────┐
   │  squeeze the  │                │  rebuild the  │
   │  image down   │                │  image back   │
   └───────────────┘                └───────────────┘

   image  x  ─────►  latent  z  ─────►  reconstruction  x̂
  (512×512×3)       (small code)        (512×512×3)
```

- **Encode:** `z = E(x)` — turn the image into a small code.
- **Decode:** `x̂ = D(z)` — turn the code back into an image.
- A *good* VAE makes `x̂` look almost exactly like `x`. The round-trip
  `D(E(x))` is nearly the identity for natural images.

Why bother? The big generative transformer never touches pixels — it works
entirely in the small latent space, which is far cheaper. The VAE is the
translator at both ends.

### The bottleneck — the single most important idea

The latent is *much* smaller than the image. That forced shrinkage is called
the **bottleneck**, and its size is the whole story for this report.

For a 512×512 image, one 8×8 patch of pixels = `3 × 8 × 8 = 192` numbers.
Each VAE compresses that patch into one latent "pixel":

```
   one 8×8 image patch          latent pixel
   = 192 numbers     ──────►    FLUX.1: 16 numbers   (192/16 = 12× squeeze)
                                FLUX.2: 32 numbers   (192/32 =  6× squeeze)
```

**FLUX.1 squeezes 12×. FLUX.2 squeezes only 6×.** FLUX.2 keeps twice as much
information per location. Hold onto that number — it is the root cause of
everything.

---

## Level 2 — What an adversarial attack is

An adversarial attack adds a **tiny, carefully-shaped perturbation** `δ` to the
image. "Tiny" means each pixel moves by at most `ε` (epsilon) — small enough
that to your eye `x` and `x + δ` look identical.

```
     x              +   δ (×10 to see it)    =    x_adv
  ┌───────┐            ┌───────┐                ┌───────┐
  │ cat   │            │ static│                │ cat   │   ← looks the same
  │ photo │            │ noise │                │ photo │      to a human
  └───────┘            └───────┘                └───────┘
```

The attack is *not* random static. It is optimized so that after the image goes
through the VAE, the output is wrecked. The recipe (PGD — Projected Gradient
Descent) is a loop:

```
repeat ~40 times:
    1. push x_adv through the VAE:  x̂ = D(E(x_adv))
    2. measure how broken the output is:  loss = || x̂ − x ||²
    3. ask calculus which direction makes it MORE broken  (the gradient)
    4. nudge δ a tiny step that way
    5. clip δ so no pixel ever exceeds ε   (stay invisible)
```

Step 3 — "ask calculus which direction" — is where the gradient (and later, the
Jacobian) enters. We'll get there.

### What we measure

The headline robustness number in this repo is **`decoded_diff_mse`**:

```
decoded_diff_mse = || D(E(x_adv)) − D(E(x)) ||²
                    \_______________________/
                     how much the OUTPUT image changed
                     when the INPUT barely changed
```

Small = robust (invisible input change → invisible output change).
Large  = fragile (invisible input change → visible output damage).

> **Why this metric and not `latent_mse`?** The latent spaces have different
> sizes (16 vs 32 channels) and different scaling, so their raw distances aren't
> comparable. `decoded_diff_mse` lives in ordinary RGB pixel space — same units
> for both models — so it's a fair head-to-head.

---

## Level 3 — The finding, in plain words

Two facts that seem to contradict each other:

```
                       clean quality        robustness
                    (recon of normal img)  (damage under attack)
   FLUX.1   (12×)        good                 GOOD  (small damage)
   FLUX.2   ( 6×)        BETTER               bad   (8–18× more damage)
```

FLUX.2 is the *better* autoencoder at its day job (reconstructing real images),
yet it falls apart under attack. That combination — **better clean quality +
worse robustness** — is the fingerprint of a **quality/robustness tradeoff**.
It tells us this isn't a broken model; it's a model tuned for a different goal.

---

## Level 4 — The intuition for *why* (no math yet)

### The bottleneck is a filter

Think of the encoder's bottleneck as a sieve. Natural image content is "big" and
structured; the adversarial perturbation is "small" and fragile, riding on the
high-frequency, low-amplitude part of the signal.

```
   FLUX.1 — narrow sieve (12×)        FLUX.2 — wide sieve (6×)
   ┌─────────────────────┐           ┌─────────────────────┐
   │ ███ image content   │ passes    │ ███ image content   │ passes
   │ ·   adversarial δ   │ BLOCKED   │ ·   adversarial δ   │ passes too
   └─────────────────────┘           └─────────────────────┘
   keeps less → discards δ           keeps more → keeps δ
```

To squeeze 12×, FLUX.1 *must* throw information away — and the cheapest things
to throw away are exactly the small, high-frequency details where the attack
hides. FLUX.2, only squeezing 6×, has room to keep that detail. That's why it
reconstructs clean images better (real detail survives) **and** why it's more
fragile (adversarial detail survives too). Same property, two consequences.

This is the same reason JPEG compression or reducing color depth can blunt
adversarial attacks: compression is an accidental defense.

### The decoder then "believes" the smuggled signal

Once the perturbation survives encoding, the latent code `z` has been nudged off
to the side. FLUX.2's decoder is high-fidelity — it was trained to render every
latent direction as crisp image detail. So when the attack nudges `z` into an
unnatural direction, the decoder faithfully paints that nonsense onto the
output. It can't tell "real detail" from "adversarial detail" — it renders both.

So the failure is a **two-step relay**:

```
   attacker          encoder lets it           decoder renders it
   adds δ    ──►     through to z      ──►      as visible damage
   (Level 2)         (bottleneck)              (high-fidelity decoder)
```

FLUX.1 breaks the relay at step 1 (the narrow sieve blocks δ). FLUX.2 lets it
run all the way to a visible failure.

---

## Level 5 — Now the math (and it's just the chain rule)

We want to express "how much does the output move when I nudge the input?" For a
tiny nudge `δ`, calculus says a function is approximately linear — its local
behavior is captured by its **Jacobian** (the matrix of all partial
derivatives; think "multi-dimensional slope").

For the full round-trip `x ↦ D(E(x))`, the chain rule gives:

```
   D(E(x + δ)) − D(E(x))   ≈   J_D · J_E · δ
   \____________________/        │    │   │
    change in output image       │    │   └─ the tiny input nudge
                                  │    └──── encoder Jacobian:
                                  │          how far z moves per unit of δ
                                  └───────── decoder Jacobian:
                                             how far the image moves per unit of z
```

Read it left to right as the relay from Level 4:

1. `δ` — the attacker's tiny pixel nudge (bounded: every entry ≤ ε).
2. `J_E · δ` — the encoder turns that into latent movement. A **strong
   bottleneck shrinks this** (it's nearly singular in the directions it throws
   away), so for FLUX.1 the adversarial part of `δ` lands in the encoder's
   "null-ish" space and barely moves `z`.
3. `J_D · (…)` — the decoder turns latent movement into image movement. A
   **high-fidelity decoder has large gain here** in many directions, because it
   was trained to make latent detail visible.

The output damage is the product of the two gains. To be robust you want the
product small. FLUX.1 wins by keeping `J_E` small in adversarial directions
(the sieve). FLUX.2 loses on both: a roomier latent means `J_E` doesn't kill the
perturbation, and a sharper decoder means `J_D` amplifies whatever gets through.

### Connecting the math back to the numbers we already have

The repo's metrics are direct, measurable stand-ins for the terms above:

| Quantity in the math | Metric in the repo | What it isolates |
|---|---|---|
| size of `δ` | `pixel_linf` (= ε), `pixel_mse` | the input budget (held equal for both models) |
| size of `J_E · δ` | `latent_mse`, `latent_linf` | encoder movement (compare *cautiously* — different latent scales) |
| size of `J_D · J_E · δ` | **`decoded_diff_mse`** | the full relay, in pixel space — **the fair metric** |
| clean `‖D(E(x)) − x‖²` | `recon_mse_orig` | day-job quality (FLUX.2 wins this) |

Notice the evidence matches the story: under attack FLUX.2's `latent_linf` is
*larger* than FLUX.1's (encoder moves further — bigger `J_E` term) **and** its
`decoded_diff_mse` blows up far more than that latent gap alone would suggest
(decoder amplifies it — bigger `J_D` term). Both Jacobian factors point the same
way.

---

## Level 6 — Why FLUX.2 was built this way (it's not a bug)

FLUX.2's VAE was **retrained from scratch** to improve the
learnability–quality–compression tradeoff for its generative transformer. That
objective rewards a latent space that is *informative and easy to learn from* —
i.e. one that preserves detail (6× instead of 12×, 32 channels instead of 16).

Nothing in that objective asks for:
- a small encoder Jacobian (Lipschitz/robustness regularization), or
- a locally flat decoder around natural images, or
- adversarial training against PGD-style perturbations.

So FLUX.2 optimized exactly what it was asked to optimize — clean fidelity and
generative learnability — and adversarial fragility came along as an unpriced
side effect. FLUX.1's robustness is likewise *accidental*: it's a free byproduct
of its tighter bottleneck, not a deliberate defense.

> **Important scoping note.** The 2×2 latent *packing* and the *BatchNorm*
> normalization that FLUX.2 uses are applied in the **generation pipeline**, not
> inside the VAE's `encode()` / `decode()` that this study attacks. So the
> mechanism here is driven by **channel count / compression ratio and the
> encoder–decoder pair**, *not* by the packing/BN. (See the corrected report.)

---

## Level 7 — One-paragraph summary you can repeat

FLUX.1's autoencoder compresses images 12× through a 16-channel bottleneck;
FLUX.2's compresses only 6× through 32 channels. A tighter bottleneck acts as a
sieve that discards the small, high-frequency signal an adversarial attack hides
in — so the perturbation never reaches FLUX.1's decoder. FLUX.2 keeps that
detail (which is why it reconstructs clean images better), so the perturbation
survives encoding, and its high-fidelity decoder then renders the corrupted
latent as visible damage. Formally, output sensitivity is the product of the
encoder and decoder Jacobians `J_D · J_E`; FLUX.1 keeps `J_E` small in
adversarial directions while FLUX.2 is larger on both factors. This is a
quality/robustness tradeoff, not a defect — FLUX.2 was optimized for fidelity
and learnability, never for adversarial stability.

---

## Where this goes next (preview of the code step)

The Jacobian story above is currently an *inference* from `decoded_diff_mse`. To
turn it into *measurement*, we can:

1. **Isolate the decoder** — perturb the latent directly (`z + η`) and measure
   `‖D(z+η) − D(z)‖ / ‖η‖`. Does FLUX.2's decoder really amplify more?
2. **Isolate the encoder** — measure `‖E(x+δ) − E(x)‖ / ‖δ‖` for random vs PGD
   directions, normalized per-channel so 16-ch and 32-ch are comparable.
3. **Frequency analysis** — check whether successful FLUX.2 perturbations really
   are the high-frequency content the sieve picture predicts.

Those three experiments would confirm (or complicate) the picture drawn here.
We'll design them in the next step.
