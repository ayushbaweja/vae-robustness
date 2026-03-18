import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from diffusers import AutoencoderKLFlux2
from sklearn.decomposition import PCA


def load_image(path: str, size: int = 512) -> torch.Tensor:
    """Load image, resize, return tensor in [-1, 1] with shape (1, 3, H, W)."""
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) or (3,H,W) in [-1,1] → (H,W,3) in [0,1]."""
    if t.dim() == 4:
        t = t[0]
    return (t.permute(1, 2, 0).cpu().float() * 0.5 + 0.5).clamp(0, 1).numpy()


def latent_to_rgb_channels(z: torch.Tensor) -> np.ndarray:
    """Visualize latent by taking first 3 channels, normalizing to [0,1].
    z: (1, C, h, w) → returns (h, w, 3) numpy array."""
    z3 = z[0, :3].cpu().float()  # (3, h, w)
    for c in range(3):
        mn, mx = z3[c].min(), z3[c].max()
        z3[c] = (z3[c] - mn) / (mx - mn + 1e-8)
    return z3.permute(1, 2, 0).numpy()


def latent_to_rgb_pca(z: torch.Tensor) -> np.ndarray:
    """Visualize latent via PCA → 3 components, normalized to [0,1].
    z: (1, C, h, w) → returns (h, w, 3) numpy array."""
    C, h, w = z.shape[1], z.shape[2], z.shape[3]
    pixels = z[0].cpu().float().reshape(C, -1).T.numpy()  # (h*w, C)
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(pixels)  # (h*w, 3)
    for c in range(3):
        mn, mx = rgb[:, c].min(), rgb[:, c].max()
        rgb[:, c] = (rgb[:, c] - mn) / (mx - mn + 1e-8)
    return rgb.reshape(h, w, 3)


# ── PGD Attack ───────────────────────────────────────────────────────────────

def pgd_attack_vae(
    vae,
    images: torch.Tensor,
    epsilon: float = 0.06,
    alpha: float = 0.01,
    num_iter: int = 40,
    loss_mode: str = "pixel",
) -> torch.Tensor:
    """PGD attack on VAE.

    loss_mode:
        "pixel"  — max ||decode(encode(x+δ)) - x||²  (gradient through full VAE)
        "latent" — max ||encode(x+δ) - encode(x)||²   (gradient through encoder only)
    """
    vae.eval()
    device = images.device

    with torch.no_grad():
        z_orig = vae.encode(images).latent_dist.mode()

    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, -1, 1) - images
    delta.requires_grad = True

    for i in range(num_iter):
        adv = images + delta

        posterior = vae.encode(adv).latent_dist
        z = posterior.mode()

        if loss_mode == "latent":
            loss = F.mse_loss(z, z_orig, reduction="sum")
        else:
            recon = vae.decode(z).sample
            loss = F.mse_loss(recon, images, reduction="sum")

        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(images + delta, -1, 1) - images
            delta.requires_grad = True

    return (images + delta).detach()


# ── Visualization ────────────────────────────────────────────────────────────

@torch.no_grad()
def visualize_attack(vae, original, adversarial, save_path, image_name=""):
    """Create a comprehensive visualization figure."""
    device = original.device

    z_orig = vae.encode(original).latent_dist.mode()
    z_adv = vae.encode(adversarial).latent_dist.mode()

    dec_orig = vae.decode(z_orig).sample
    dec_adv = vae.decode(z_adv).sample

    img_orig = tensor_to_numpy_image(original)
    img_adv = tensor_to_numpy_image(adversarial)
    perturbation = np.clip(np.abs(img_adv - img_orig) * 10, 0, 1)

    lat_ch_orig = latent_to_rgb_channels(z_orig)
    lat_ch_adv = latent_to_rgb_channels(z_adv)
    lat_ch_diff_raw = (z_adv[0, :3] - z_orig[0, :3]).abs().cpu().float()
    lat_ch_diff = lat_ch_diff_raw * 10
    for c in range(3):
        mn, mx = lat_ch_diff[c].min(), lat_ch_diff[c].max()
        lat_ch_diff[c] = (lat_ch_diff[c] - mn) / (mx - mn + 1e-8)
    lat_ch_diff = lat_ch_diff.permute(1, 2, 0).numpy()

    lat_pca_orig = latent_to_rgb_pca(z_orig)
    lat_pca_adv = latent_to_rgb_pca(z_adv)
    lat_pca_diff = np.clip(np.abs(lat_pca_adv - lat_pca_orig) * 10, 0, 1)

    dec_orig_np = tensor_to_numpy_image(dec_orig)
    dec_adv_np = tensor_to_numpy_image(dec_adv)
    dec_diff = np.clip(np.abs(dec_adv_np - dec_orig_np) * 10, 0, 1)

    pixel_mse = F.mse_loss(adversarial, original).item()
    pixel_linf = (adversarial - original).abs().max().item()
    latent_mse = F.mse_loss(z_adv, z_orig).item()
    latent_linf = (z_adv - z_orig).abs().max().item()
    recon_mse_orig = F.mse_loss(dec_orig, original).item()
    recon_mse_adv = F.mse_loss(dec_adv, original).item()
    dec_diff_mse = F.mse_loss(dec_adv, dec_orig).item()

    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.suptitle(
        f"{image_name}\n"
        f"Pixel: MSE={pixel_mse:.6f}, L∞={pixel_linf:.4f} | "
        f"Latent: MSE={latent_mse:.4f}, L∞={latent_linf:.4f}\n"
        f"Recon MSE (orig)={recon_mse_orig:.6f}, Recon MSE (adv→orig)={recon_mse_adv:.6f}, "
        f"Dec diff MSE={dec_diff_mse:.6f}",
        fontsize=12, y=0.98,
    )

    titles = [
        ["Original", "Adversarial", "Perturbation (×10)"],
        ["Latent (ch 0-2) Original", "Latent (ch 0-2) Adversarial", "Latent Ch Diff (×10)"],
        ["Latent (PCA) Original", "Latent (PCA) Adversarial", "Latent PCA Diff (×10)"],
        ["Decoded Original", "Decoded Adversarial", "Decoded Diff (×10)"],
    ]
    images_grid = [
        [img_orig, img_adv, perturbation],
        [lat_ch_orig, lat_ch_adv, lat_ch_diff],
        [lat_pca_orig, lat_pca_adv, lat_pca_diff],
        [dec_orig_np, dec_adv_np, dec_diff],
    ]

    for r in range(4):
        for c in range(3):
            axes[r][c].imshow(images_grid[r][c])
            axes[r][c].set_title(titles[r][c], fontsize=11)
            axes[r][c].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "pixel_mse": pixel_mse,
        "pixel_linf": pixel_linf,
        "latent_mse": latent_mse,
        "latent_linf": latent_linf,
        "recon_mse_orig": recon_mse_orig,
        "recon_mse_adv_vs_orig": recon_mse_adv,
        "decoded_diff_mse": dec_diff_mse,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse, json

    parser = argparse.ArgumentParser(description="PGD attack on FLUX.2 VAE")
    parser.add_argument("--input_dir", type=str, default="resources/test-images")
    parser.add_argument("--output_dir", type=str, default="results/flux2_pgd")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-schnell",
                        help="HF model ID for FLUX.2 (requires subfolder='vae')")
    parser.add_argument("--epsilon", type=float, default=0.06)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--num_iter", type=int, default=40)
    parser.add_argument("--image_size", type=int, default=512,
                        help="Must be divisible by 16")
    parser.add_argument("--loss", type=str, default="pixel", choices=["pixel", "latent"],
                        help="'pixel': max recon error through full VAE, "
                             "'latent': max latent displacement (encoder only)")
    args = parser.parse_args()

    assert args.image_size % 16 == 0, f"image_size must be divisible by 16, got {args.image_size}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load FLUX.2 VAE (32 latent channels)
    print(f"Loading FLUX.2 VAE from {args.model_id} ...")
    vae = AutoencoderKLFlux2.from_pretrained(args.model_id, subfolder="vae")
    vae = vae.to(device).float().eval()
    print(f"VAE loaded. Latent channels: {vae.config.latent_channels}")

    input_path = Path(args.input_dir)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG")
    image_files = sorted(sum([list(input_path.glob(e)) for e in exts], []))
    print(f"Found {len(image_files)} images in {input_path}")

    if not image_files:
        return

    output_path = Path(args.output_dir) / f"eps_{args.epsilon}_{args.loss}"
    output_path.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        img_tensor = load_image(str(img_path), args.image_size).to(device)

        adv_tensor = pgd_attack_vae(
            vae, img_tensor,
            epsilon=args.epsilon, alpha=args.alpha, num_iter=args.num_iter,
            loss_mode=args.loss,
        )

        adv_pil = Image.fromarray(
            (tensor_to_numpy_image(adv_tensor) * 255).astype(np.uint8)
        )
        adv_pil.save(output_path / f"{img_path.stem}_adv.png")

        metrics = visualize_attack(
            vae, img_tensor, adv_tensor,
            save_path=output_path / f"{img_path.stem}_visualization.png",
            image_name=img_path.name,
        )
        metrics["image"] = img_path.name
        all_metrics.append(metrics)

        print(f"  Pixel MSE={metrics['pixel_mse']:.6f}, L∞={metrics['pixel_linf']:.4f}")
        print(f"  Latent MSE={metrics['latent_mse']:.4f}, L∞={metrics['latent_linf']:.4f}")
        print(f"  Decoded diff MSE={metrics['decoded_diff_mse']:.6f}")

    summary = {
        "config": vars(args),
        "per_image": all_metrics,
        "average": {
            k: round(np.mean([m[k] for m in all_metrics]), 6)
            for k in all_metrics[0] if k != "image"
        },
    }
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    main()
