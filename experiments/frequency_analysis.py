"""Experiment 3 -- Frequency analysis of perturbations.

Part A: radial power spectrum of the *adversarial* delta. We regenerate the
PGD-pixel attack in-memory (so delta keeps full precision) and recover
delta = x_adv - x, take its 2D FFT, average magnitude over channels, square to
power, and radially average. Tests whether FLUX.2's effective delta carries more
high-frequency power than FLUX.1's.

Part B (the cleaner probe): band-survival test. We synthesize band-limited unit
perturbations whose FFT energy lies only in a single radial frequency ring,
normalized to a fixed pixel L2 norm, and measure how much each band survives the
round-trip:

    survival(band) = ||D(E(x + d_band)) - D(E(x))||_2 / ||d_band||_2

The same d_band is used for both models (deterministic per image/band), so the
comparison is fair. Hypothesis: FLUX.1's survival drops faster at high frequency
(stronger low-pass sieve), FLUX.2 stays higher.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import vae_common as vc


# ── frequency grids ───────────────────────────────────────────────────────────

def radial_freq_grid(h, w):
    """Radial spatial frequency (cycles/pixel) for an h x w FFT, unshifted."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)


def radial_average(vals2d, fr2d, nbins=64, fmax=0.5):
    bins = np.linspace(0.0, fmax, nbins + 1)
    idx = np.digitize(fr2d.ravel(), bins) - 1
    v = vals2d.ravel()
    out = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.any():
            out[b] = v[m].mean()
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, out


def band_edges(nbands, h):
    """Log-spaced radial frequency ring edges from ~1/H up to Nyquist (0.5)."""
    return np.logspace(np.log10(1.0 / h), np.log10(0.5), nbands + 1)


def make_band_delta(h, w, lo, hi, fr2d, target_l2, device, generator):
    """Band-limited real perturbation (3,h,w): FFT energy only in [lo, hi)."""
    mask = torch.from_numpy(((fr2d >= lo) & (fr2d < hi)).astype(np.float32)).to(device)
    noise = torch.randn(3, h, w, device=device, generator=generator)
    spec = torch.fft.fft2(noise) * mask
    d = torch.fft.ifft2(spec).real
    n = d.flatten().norm()
    if n < 1e-12:
        return None
    d = d / n * target_l2
    return d.unsqueeze(0)  # (1,3,h,w)


# ── Part A: spectrum of adversarial delta ─────────────────────────────────────

def run_part_a(key, image_files, args, device):
    vae = vc.load_vae(key, device)
    meta = vc.model_metadata(key, vae)
    print(f"\n[A] {meta['label']} -- adversarial delta spectrum")
    h = w = args.image_size
    fr = radial_freq_grid(h, w)
    profiles = []
    for img_path in image_files:
        x = vc.load_image(str(img_path), args.image_size).to(device)
        adv = vc.pgd_attack_vae(
            vae, x, epsilon=args.eps_spectrum, alpha=args.alpha,
            num_iter=args.num_iter, loss_mode="pixel",
        )
        delta = (adv - x)[0].detach().cpu().numpy()  # (3,h,w)
        mag = np.abs(np.fft.fft2(delta, axes=(-2, -1))).mean(axis=0)  # avg over ch
        power = mag ** 2
        centers, prof = radial_average(power, fr, nbins=args.nbins)
        profiles.append(prof)
        print(f"  {img_path.name}")
    profiles = np.array(profiles)
    avg_profile = np.nanmean(profiles, axis=0)
    del vae
    torch.cuda.empty_cache()
    return {
        "model": meta,
        "freq_centers": centers.tolist(),
        "radial_power": avg_profile.tolist(),
        "eps_spectrum": args.eps_spectrum,
    }


# ── Part B: band-survival ─────────────────────────────────────────────────────

def run_part_b(key, image_files, args, device):
    vae = vc.load_vae(key, device)
    meta = vc.model_metadata(key, vae)
    print(f"\n[B] {meta['label']} -- band survival")
    h = w = args.image_size
    fr = radial_freq_grid(h, w)
    edges = band_edges(args.nbands, h)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric ring centers
    target_l2 = args.band_rms * (3 * h * w) ** 0.5

    per_image = []
    for ii, img_path in enumerate(image_files):
        x = vc.load_image(str(img_path), args.image_size).to(device)
        with torch.no_grad():
            dec0 = vc.decode(vae, vc.encode(vae, x))
        surv = []
        for bi in range(args.nbands):
            # deterministic per (image, band) -> identical d_band across models
            gen = torch.Generator(device=device).manual_seed(
                args.seed + ii * 1000 + bi
            )
            d = make_band_delta(h, w, edges[bi], edges[bi + 1], fr,
                                target_l2, device, gen)
            if d is None:
                surv.append(float("nan"))
                continue
            with torch.no_grad():
                dec = vc.decode(vae, vc.encode(vae, x + d))
            surv.append(vc.l2(dec - dec0) / vc.l2(d))
        per_image.append({"image": img_path.name, "survival": surv})
        print(f"  {img_path.name}: " +
              " ".join(f"{s:.3f}" for s in surv))

    surv_arr = np.array([p["survival"] for p in per_image], dtype=np.float64)
    avg = np.nanmean(surv_arr, axis=0).tolist()
    std = np.nanstd(surv_arr, axis=0).tolist()
    del vae
    torch.cuda.empty_cache()
    return {
        "model": meta,
        "band_centers": centers.tolist(),
        "band_edges": edges.tolist(),
        "survival_mean": avg,
        "survival_std": std,
        "band_rms": args.band_rms,
        "target_l2": target_l2,
        "per_image": per_image,
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_part_a(res_a, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"flux1": "tab:blue", "flux2": "tab:red"}
    for key in vc.MODEL_ORDER:
        r = res_a[key]
        c = np.array(r["freq_centers"])
        p = np.array(r["radial_power"])
        # normalize each curve so total power = 1 (compare shape, not magnitude)
        p_norm = p / np.nansum(p)
        ax.plot(c, p_norm, color=colors[key], label=r["model"]["label"])
    ax.set_yscale("log")
    ax.set_xlabel("spatial frequency (cycles/pixel)")
    ax.set_ylabel("normalized radial power (log)")
    ax.set_title("Exp 3A - Radial spectrum of adversarial delta")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_part_b(res_b, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"flux1": "tab:blue", "flux2": "tab:red"}
    for key in vc.MODEL_ORDER:
        r = res_b[key]
        c = np.array(r["band_centers"])
        m = np.array(r["survival_mean"])
        s = np.array(r["survival_std"])
        ax.errorbar(c, m, yerr=s, marker="o", capsize=3,
                    color=colors[key], label=r["model"]["label"])
    ax.set_xscale("log")
    ax.set_xlabel("band center frequency (cycles/pixel, log)")
    ax.set_ylabel("round-trip survival  ||D(E(x+d))-D(E(x))|| / ||d||")
    ax.set_title("Exp 3B - Band survival through the round-trip")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Experiment 3: frequency analysis")
    p.add_argument("--input_dir", default="resources/test-images")
    p.add_argument("--output_dir", default="results/jacobian/freq")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--part", choices=["a", "b", "both"], default="both")
    # Part A
    p.add_argument("--eps_spectrum", type=float, default=0.06)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--num_iter", type=int, default=40)
    p.add_argument("--nbins", type=int, default=64)
    # Part B
    p.add_argument("--nbands", type=int, default=8)
    p.add_argument("--band_rms", type=float, default=0.03,
                   help="per-pixel RMS of each band perturbation; "
                        "||d|| = band_rms * sqrt(3*H*W)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models", nargs="+", default=vc.MODEL_ORDER)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_files = vc.list_images(args.input_dir, args.limit)
    print(f"Device: {device} | {len(image_files)} images from {args.input_dir}")
    out_dir = Path(args.output_dir)

    cfg = {
        "input_dir": args.input_dir, "image_size": args.image_size,
        "n_images": len(image_files), "seed": args.seed, "device": device,
        "eps_spectrum": args.eps_spectrum, "alpha": args.alpha,
        "num_iter": args.num_iter, "nbins": args.nbins,
        "nbands": args.nbands, "band_rms": args.band_rms,
    }

    if args.part in ("a", "both"):
        res_a = {k: run_part_a(k, image_files, args, device) for k in args.models}
        vc.save_json({"experiment": "freq_part_a", "config": cfg, "results": res_a},
                     out_dir / "part_a_spectrum.json")
        if len(args.models) == 2:
            plot_part_a(res_a, out_dir / "part_a_spectrum.png")

    if args.part in ("b", "both"):
        res_b = {k: run_part_b(k, image_files, args, device) for k in args.models}
        vc.save_json({"experiment": "freq_part_b", "config": cfg, "results": res_b},
                     out_dir / "part_b_survival.json")
        if len(args.models) == 2:
            plot_part_b(res_b, out_dir / "part_b_survival.png")


if __name__ == "__main__":
    main()
