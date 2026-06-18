"""Experiment 1 -- Decoder sensitivity (isolate J_D).

Question: given equal *normalized* latent movement, does FLUX.2's decoder turn it
into more pixel-space change than FLUX.1's?

For each image we perturb the latent directly (z + eta) and measure the pixel
response D(z+eta) - D(z). The perturbation budget is expressed in *std-normalized*
latent units (eta[:, c] = r * s_c * noise) so the 16-channel and 32-channel
latent spaces are compared on equal footing.

We report, per relative magnitude r:
  resp        = ||D(z+eta) - dec0||_2          (pixel L2)
  gain        = resp / ||eta_norm||_2          (eta_norm = eta / s_c, std-units)
  decoded_mse = MSE(D(z+eta), dec0)
for random directions (N_rand samples) and the worst-case direction found by
power iteration (top singular value of J_D in std-normalized coordinates).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import vae_common as vc


def random_direction_stats(vae, z, dec0, s_c, r, n_rand, generator):
    """N_rand random std-normalized latent perturbations at relative size r."""
    resps, gains, mses = [], [], []
    N = z.numel()
    with torch.no_grad():
        for _ in range(n_rand):
            noise = torch.randn(z.shape, device=z.device, generator=generator)
            eta = noise * (r * s_c)              # per-channel scaling
            eta_norm = eta / s_c                 # == r * noise  (std-units)
            dec = vc.decode(vae, z + eta)
            resp = vc.l2(dec - dec0)
            denom = vc.l2(eta_norm)
            resps.append(resp)
            gains.append(resp / denom)
            mses.append(float(((dec - dec0) ** 2).mean().item()))
    return resps, gains, mses


def worst_case_gain(vae, z, dec0, s_c, r, n_steps, generator):
    """Power iteration for the top decoder gain in std-normalized coordinates.

    Work in u = eta / s_c (std-units). Maximize ||D(z + u*s_c) - dec0||^2 at a
    fixed budget ||u|| = B that matches the random case (B = r * sqrt(N)).
    Each step replaces the direction with the gradient (power method on J_D^T J_D),
    re-normalized to the budget.
    """
    N = z.numel()
    B = r * (N ** 0.5)
    u = torch.randn(z.shape, device=z.device, generator=generator)
    u = u / u.flatten().norm() * B
    for _ in range(n_steps):
        u = u.detach().requires_grad_(True)
        eta = u * s_c
        dec = vc.decode(vae, z + eta)
        loss = ((dec - dec0) ** 2).sum()
        grad = torch.autograd.grad(loss, u)[0]
        with torch.no_grad():
            g_norm = grad.flatten().norm()
            if g_norm < 1e-12:
                break
            u = grad / g_norm * B
    with torch.no_grad():
        eta = u * s_c
        dec = vc.decode(vae, z + eta)
        resp = vc.l2(dec - dec0)
        gain = resp / vc.l2(u)              # ||u|| == B
        mse = float(((dec - dec0) ** 2).mean().item())
    return resp, gain, mse


def run_model(key, image_files, args, device):
    vae = vc.load_vae(key, device)
    meta = vc.model_metadata(key, vae)
    print(f"\n=== {meta['label']} ({meta['vae_class']}, {meta['latent_channels']}ch) ===")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    per_image = []
    for img_path in image_files:
        x = vc.load_image(str(img_path), args.image_size).to(device)
        with torch.no_grad():
            z = vc.encode(vae, x)
            dec0 = vc.decode(vae, z)
            s_c = vc.channel_std(z)
        rec = {"image": img_path.name, "r": {}}
        for r in args.r_values:
            resps, gains, mses = random_direction_stats(
                vae, z, dec0, s_c, r, args.n_rand, gen
            )
            wc_resp, wc_gain, wc_mse = worst_case_gain(
                vae, z, dec0, s_c, r, args.power_steps, gen
            )
            rec["r"][str(r)] = {
                "random": {
                    "resp": vc.mean_std(resps),
                    "gain": vc.mean_std(gains),
                    "decoded_mse": vc.mean_std(mses),
                },
                "worst_case": {
                    "resp": wc_resp,
                    "gain": wc_gain,
                    "decoded_mse": wc_mse,
                },
            }
            print(
                f"  {img_path.name:18s} r={r:<5} "
                f"rand_gain={np.mean(gains):.4f} wc_gain={wc_gain:.4f}"
            )
        per_image.append(rec)

    # aggregate over images
    avg = {}
    for r in args.r_values:
        rk = str(r)
        rand_gain = [im["r"][rk]["random"]["gain"]["mean"] for im in per_image]
        rand_resp = [im["r"][rk]["random"]["resp"]["mean"] for im in per_image]
        rand_mse = [im["r"][rk]["random"]["decoded_mse"]["mean"] for im in per_image]
        wc_gain = [im["r"][rk]["worst_case"]["gain"] for im in per_image]
        wc_resp = [im["r"][rk]["worst_case"]["resp"] for im in per_image]
        wc_mse = [im["r"][rk]["worst_case"]["decoded_mse"] for im in per_image]
        avg[rk] = {
            "random": {
                "gain": vc.mean_std(rand_gain),
                "resp": vc.mean_std(rand_resp),
                "decoded_mse": vc.mean_std(rand_mse),
            },
            "worst_case": {
                "gain": vc.mean_std(wc_gain),
                "resp": vc.mean_std(wc_resp),
                "decoded_mse": vc.mean_std(wc_mse),
            },
        }

    del vae
    torch.cuda.empty_cache()
    return {"model": meta, "per_image": per_image, "average": avg}


def make_plot(results, args, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"flux1": "tab:blue", "flux2": "tab:red"}
    for kind, ax in zip(["random", "worst_case"], axes):
        for key in vc.MODEL_ORDER:
            res = results[key]
            rs = args.r_values
            means = [res["average"][str(r)][kind]["gain"]["mean"] for r in rs]
            stds = [res["average"][str(r)][kind]["gain"]["std"] for r in rs]
            ax.errorbar(
                rs, means, yerr=stds, marker="o", capsize=3,
                color=colors[key], label=res["model"]["label"],
            )
        ax.set_xlabel("relative latent magnitude r (fraction of per-channel std)")
        ax.set_ylabel("decoder gain  ||D(z+eta)-D(z)||_2 / ||eta||_std")
        ax.set_title(f"Decoder gain ({kind.replace('_', '-')})")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Experiment 1 - Decoder sensitivity (J_D)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Experiment 1: decoder sensitivity (J_D)")
    p.add_argument("--input_dir", default="resources/test-images")
    p.add_argument("--output_dir", default="results/jacobian/decoder")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--n_rand", type=int, default=16)
    p.add_argument("--power_steps", type=int, default=10)
    p.add_argument("--r_values", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.4])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models", nargs="+", default=vc.MODEL_ORDER)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_files = vc.list_images(args.input_dir, args.limit)
    print(f"Device: {device} | {len(image_files)} images from {args.input_dir}")

    results = {}
    for key in args.models:
        results[key] = run_model(key, image_files, args, device)

    out_dir = Path(args.output_dir)
    summary = {
        "experiment": "decoder_sensitivity",
        "config": {
            "input_dir": args.input_dir,
            "image_size": args.image_size,
            "n_rand": args.n_rand,
            "power_steps": args.power_steps,
            "r_values": args.r_values,
            "n_images": len(image_files),
            "seed": args.seed,
            "device": device,
        },
        "results": results,
    }
    vc.save_json(summary, out_dir / "summary.json")
    if len(args.models) == 2:
        make_plot(results, args, out_dir / "decoder_gain.png")


if __name__ == "__main__":
    main()
