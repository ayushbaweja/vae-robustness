"""Experiment 2 -- Encoder sensitivity (isolate J_E).

Question: does FLUX.1's tighter bottleneck shrink adversarial pixel perturbations
more than FLUX.2's?

We measure *normalized* latent movement per unit pixel nudge:

    dz_norm     = ||(E(x+d) - E(x)) / s_c||_2     (per-channel divide, then L2)
    encoder_gain = dz_norm / ||d||_2

for three direction types at a fixed L-inf input budget epsilon:
  * random      -- d uniform in [-eps, eps], x+d clamped to [-1, 1];
  * pgd_latent  -- worst-case encoder direction (max ||E(x+d)-E(x)||^2);
  * pgd_pixel   -- the full round-trip adversarial d (max ||D(E(x+d))-x||^2).

For the pgd_pixel direction we also record the round-trip pixel response
(||D(E(x+d)) - D(E(x))|| and decoded_diff_mse) so it can be cross-checked
against (encoder gain) x (decoder gain) from Experiment 1.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import vae_common as vc


def dz_norm(vae, x, z0, s_c, adv):
    """Std-normalized latent movement for input adv vs clean x (z0=E(x))."""
    with torch.no_grad():
        z = vc.encode(vae, adv)
    return vc.l2((z - z0) / s_c)


def run_model(key, image_files, args, device):
    vae = vc.load_vae(key, device)
    meta = vc.model_metadata(key, vae)
    print(f"\n=== {meta['label']} ({meta['vae_class']}, {meta['latent_channels']}ch) ===")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    per_image = []
    for img_path in image_files:
        x = vc.load_image(str(img_path), args.image_size).to(device)
        with torch.no_grad():
            z0 = vc.encode(vae, x)
            dec0 = vc.decode(vae, z0)
            s_c = vc.channel_std(z0)
        rec = {"image": img_path.name, "eps": {}}
        for eps in args.eps_values:
            # --- random directions ---
            rand_gains, rand_dz, rand_dnorm = [], [], []
            with torch.no_grad():
                for _ in range(args.n_rand):
                    d = torch.empty_like(x).uniform_(-eps, eps, generator=gen)
                    adv = torch.clamp(x + d, -1, 1)
                    d_eff = adv - x
                    dz = dz_norm(vae, x, z0, s_c, adv)
                    dn = vc.l2(d_eff)
                    rand_dz.append(dz)
                    rand_dnorm.append(dn)
                    rand_gains.append(dz / dn)

            # --- PGD-latent (worst-case encoder direction) ---
            adv_lat = vc.pgd_attack_vae(
                vae, x, epsilon=eps, alpha=args.alpha,
                num_iter=args.num_iter, loss_mode="latent",
            )
            d_lat = (adv_lat - x)
            dz_lat = dz_norm(vae, x, z0, s_c, adv_lat)
            dnorm_lat = vc.l2(d_lat)

            # --- PGD-pixel (full round-trip adversary) ---
            adv_pix = vc.pgd_attack_vae(
                vae, x, epsilon=eps, alpha=args.alpha,
                num_iter=args.num_iter, loss_mode="pixel",
            )
            d_pix = (adv_pix - x)
            dz_pix = dz_norm(vae, x, z0, s_c, adv_pix)
            dnorm_pix = vc.l2(d_pix)
            with torch.no_grad():
                dec_adv = vc.decode(vae, vc.encode(vae, adv_pix))
            roundtrip_l2 = vc.l2(dec_adv - dec0)
            decoded_diff_mse = float(((dec_adv - dec0) ** 2).mean().item())

            rec["eps"][str(eps)] = {
                "random": {
                    "encoder_gain": vc.mean_std(rand_gains),
                    "dz_norm": vc.mean_std(rand_dz),
                    "delta_l2": vc.mean_std(rand_dnorm),
                },
                "pgd_latent": {
                    "encoder_gain": dz_lat / dnorm_lat,
                    "dz_norm": dz_lat,
                    "delta_l2": dnorm_lat,
                },
                "pgd_pixel": {
                    "encoder_gain": dz_pix / dnorm_pix,
                    "dz_norm": dz_pix,
                    "delta_l2": dnorm_pix,
                    "roundtrip_l2": roundtrip_l2,
                    "roundtrip_gain": roundtrip_l2 / dnorm_pix,
                    "decoded_diff_mse": decoded_diff_mse,
                },
            }
            print(
                f"  {img_path.name:18s} eps={eps:<5} "
                f"rand_g={np.mean(rand_gains):.4f} "
                f"pgdL_g={dz_lat / dnorm_lat:.4f} "
                f"pgdP_g={dz_pix / dnorm_pix:.4f} "
                f"dec_mse={decoded_diff_mse:.5f}"
            )
        per_image.append(rec)

    # aggregate over images. For "random" each per-image entry is a mean_std
    # dict (average its mean); for the PGD directions it is a scalar.
    def per_image_scalar(im_block, field):
        v = im_block[field]
        return v["mean"] if isinstance(v, dict) else v

    avg = {}
    for eps in args.eps_values:
        ek = str(eps)
        block = {}
        for dirtype in ["random", "pgd_latent", "pgd_pixel"]:
            gains = [per_image_scalar(im["eps"][ek][dirtype], "encoder_gain")
                     for im in per_image]
            dzs = [per_image_scalar(im["eps"][ek][dirtype], "dz_norm")
                   for im in per_image]
            block[dirtype] = {
                "encoder_gain": vc.mean_std(gains),
                "dz_norm": vc.mean_std(dzs),
            }
        # extra round-trip stats for pgd_pixel
        block["pgd_pixel"]["roundtrip_gain"] = vc.mean_std(
            [im["eps"][ek]["pgd_pixel"]["roundtrip_gain"] for im in per_image]
        )
        block["pgd_pixel"]["decoded_diff_mse"] = vc.mean_std(
            [im["eps"][ek]["pgd_pixel"]["decoded_diff_mse"] for im in per_image]
        )
        avg[ek] = block

    del vae
    torch.cuda.empty_cache()
    return {"model": meta, "per_image": per_image, "average": avg}


def make_plot(results, args, out_path):
    dirtypes = ["random", "pgd_latent", "pgd_pixel"]
    fig, axes = plt.subplots(1, len(dirtypes), figsize=(15, 5), sharey=True)
    colors = {"flux1": "tab:blue", "flux2": "tab:red"}
    for dirtype, ax in zip(dirtypes, axes):
        for key in vc.MODEL_ORDER:
            res = results[key]
            eps = args.eps_values
            means = [res["average"][str(e)][dirtype]["encoder_gain"]["mean"] for e in eps]
            stds = [res["average"][str(e)][dirtype]["encoder_gain"]["std"] for e in eps]
            ax.errorbar(eps, means, yerr=stds, marker="o", capsize=3,
                        color=colors[key], label=res["model"]["label"])
        ax.set_xlabel("epsilon (L-inf pixel budget)")
        ax.set_title(dirtype)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("encoder gain  ||dz||_std / ||d||_2")
    fig.suptitle("Experiment 2 - Encoder sensitivity (J_E, std-normalized)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Experiment 2: encoder sensitivity (J_E)")
    p.add_argument("--input_dir", default="resources/test-images")
    p.add_argument("--output_dir", default="results/jacobian/encoder")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--n_rand", type=int, default=16)
    p.add_argument("--eps_values", type=float, nargs="+", default=[0.02, 0.06, 0.2])
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--num_iter", type=int, default=40)
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
        "experiment": "encoder_sensitivity",
        "config": {
            "input_dir": args.input_dir,
            "image_size": args.image_size,
            "n_rand": args.n_rand,
            "eps_values": args.eps_values,
            "alpha": args.alpha,
            "num_iter": args.num_iter,
            "n_images": len(image_files),
            "seed": args.seed,
            "device": device,
        },
        "results": results,
    }
    vc.save_json(summary, out_dir / "summary.json")
    if len(args.models) == 2:
        make_plot(results, args, out_dir / "encoder_gain.png")


if __name__ == "__main__":
    main()
