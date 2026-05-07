import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, AutoencoderKLCogVideoX, AutoencoderKLFlux2, AutoencoderKLLTXVideo
from sklearn.decomposition import PCA


IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    image_size: int
    is_video: bool
    loader: type
    subfolder: str = "vae"


MODEL_SPECS = {
    "sd15": ModelSpec(
        name="sd15",
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        image_size=512,
        is_video=False,
        loader=AutoencoderKL,
    ),
    "flux1": ModelSpec(
        name="flux1",
        model_id="black-forest-labs/FLUX.1-schnell",
        image_size=512,
        is_video=False,
        loader=AutoencoderKL,
    ),
    "flux2": ModelSpec(
        name="flux2",
        model_id="black-forest-labs/FLUX.2-dev",
        image_size=512,
        is_video=False,
        loader=AutoencoderKLFlux2,
    ),
    "cogvideox": ModelSpec(
        name="cogvideox",
        model_id="THUDM/CogVideoX-2b",
        image_size=480,
        is_video=True,
        loader=AutoencoderKLCogVideoX,
    ),
    "ltx": ModelSpec(
        name="ltx",
        model_id="Lightricks/LTX-Video",
        image_size=512,
        is_video=True,
        loader=AutoencoderKLLTXVideo,
    ),
}


def load_image(path: str, size: int, is_video: bool) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if is_video:
        tensor = tensor.unsqueeze(2)
    return tensor


def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 5:
        t = t[0, :, 0]
    elif t.dim() == 4:
        t = t[0]
    return (t.permute(1, 2, 0).cpu().float() * 0.5 + 0.5).clamp(0, 1).numpy()


def latent_to_rgb_channels(z: torch.Tensor) -> np.ndarray:
    if z.dim() == 5:
        z3 = z[0, :3, 0].cpu().float()
    else:
        z3 = z[0, :3].cpu().float()
    for c in range(3):
        mn, mx = z3[c].min(), z3[c].max()
        z3[c] = (z3[c] - mn) / (mx - mn + 1e-8)
    return z3.permute(1, 2, 0).numpy()


def latent_to_rgb_pca(z: torch.Tensor) -> np.ndarray:
    if z.dim() == 5:
        pixels = z[0, :, 0].cpu().float()
        c, h, w = pixels.shape
    else:
        pixels = z[0].cpu().float()
        c, h, w = pixels.shape
    rgb = PCA(n_components=3).fit_transform(pixels.reshape(c, -1).T.numpy())
    for idx in range(3):
        mn, mx = rgb[:, idx].min(), rgb[:, idx].max()
        rgb[:, idx] = (rgb[:, idx] - mn) / (mx - mn + 1e-8)
    return rgb.reshape(h, w, 3)


def encode_mode(vae, images: torch.Tensor) -> torch.Tensor:
    return vae.encode(images).latent_dist.mode()


def decode_sample(vae, latents: torch.Tensor) -> torch.Tensor:
    return vae.decode(latents).sample


def evaluate_metrics(vae, original: torch.Tensor, adversarial: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        z_orig = encode_mode(vae, original)
        z_adv = encode_mode(vae, adversarial)
        dec_orig = decode_sample(vae, z_orig)
        dec_adv = decode_sample(vae, z_adv)

    return {
        "pixel_mse": F.mse_loss(adversarial, original).item(),
        "pixel_linf": (adversarial - original).abs().max().item(),
        "latent_mse": F.mse_loss(z_adv, z_orig).item(),
        "latent_linf": (z_adv - z_orig).abs().max().item(),
        "recon_mse_orig": F.mse_loss(dec_orig, original).item(),
        "recon_mse_adv_vs_orig": F.mse_loss(dec_adv, original).item(),
        "decoded_diff_mse": F.mse_loss(dec_adv, dec_orig).item(),
    }


def attack_objective(
    vae,
    original: torch.Tensor,
    z_orig: torch.Tensor,
    dec_orig: torch.Tensor,
    adversarial: torch.Tensor,
    objective: str,
) -> torch.Tensor:
    z_adv = encode_mode(vae, adversarial)

    if objective == "latent":
        return F.mse_loss(z_adv, z_orig, reduction="sum")

    dec_adv = decode_sample(vae, z_adv)
    if objective == "recon":
        return F.mse_loss(dec_adv, original, reduction="sum")
    if objective == "decoded":
        return F.mse_loss(dec_adv, dec_orig, reduction="sum")
    raise ValueError(f"Unknown objective: {objective}")


def apgd_attack_vae(
    vae,
    images: torch.Tensor,
    epsilon: float,
    num_iter: int,
    objective: str,
    restarts: int = 2,
) -> tuple[torch.Tensor, float]:
    vae.eval()
    with torch.no_grad():
        z_orig = encode_mode(vae, images)
        dec_orig = decode_sample(vae, z_orig)

    best_adv = images.clone()
    best_score = float("-inf")
    checkpoint = max(5, num_iter // 6)

    for _ in range(restarts):
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        delta = torch.clamp(images + delta, -1, 1) - images
        step = max(epsilon / 4, 1e-4)
        prev_grad = None
        plateau_count = 0
        running_best = float("-inf")

        for itr in range(num_iter):
            delta.requires_grad_(True)
            adv = images + delta
            loss = attack_objective(vae, images, z_orig, dec_orig, adv, objective)
            grad = torch.autograd.grad(loss, delta)[0]

            with torch.no_grad():
                if prev_grad is not None:
                    grad = 0.75 * grad + 0.25 * prev_grad
                delta = delta + step * grad.sign()
                delta = torch.clamp(delta, -epsilon, epsilon)
                delta = torch.clamp(images + delta, -1, 1) - images
                prev_grad = grad.detach()

                score = loss.item()
                if score > running_best + 1e-12:
                    running_best = score
                    plateau_count = 0
                else:
                    plateau_count += 1

                if (itr + 1) % checkpoint == 0 and plateau_count >= checkpoint // 2:
                    step = max(step / 2, 1e-4)
                    plateau_count = 0

        candidate = (images + delta).detach()
        candidate_metrics = evaluate_metrics(vae, images, candidate)
        candidate_score = candidate_metrics["decoded_diff_mse"]
        if candidate_score > best_score:
            best_score = candidate_score
            best_adv = candidate

    return best_adv, best_score


def square_attack_vae(
    vae,
    images: torch.Tensor,
    epsilon: float,
    num_iter: int,
    objective: str,
) -> tuple[torch.Tensor, float]:
    vae.eval()
    with torch.no_grad():
        z_orig = encode_mode(vae, images)
        dec_orig = decode_sample(vae, z_orig)

    best_adv = images.clone()
    best_score = evaluate_metrics(vae, images, best_adv)["decoded_diff_mse"]
    current_objective = attack_objective(vae, images, z_orig, dec_orig, best_adv, objective).item()
    delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, -1, 1) - images

    spatial_h = images.shape[-2]
    spatial_w = images.shape[-1]
    min_side = min(spatial_h, spatial_w)

    for itr in range(num_iter):
        p = max(0.03, 0.4 * (1 - itr / max(num_iter - 1, 1)))
        side = max(1, int(round(np.sqrt(p) * min_side)))
        top = np.random.randint(0, spatial_h - side + 1)
        left = np.random.randint(0, spatial_w - side + 1)

        proposal = delta.clone()
        patch_noise = torch.empty_like(proposal[..., top : top + side, left : left + side]).uniform_(-epsilon, epsilon)
        proposal[..., top : top + side, left : left + side] = patch_noise
        proposal = torch.clamp(proposal, -epsilon, epsilon)
        proposal = torch.clamp(images + proposal, -1, 1) - images
        adv = (images + proposal).detach()

        with torch.no_grad():
            score = attack_objective(vae, images, z_orig, dec_orig, adv, objective).item()
        if score >= current_objective:
            delta = proposal
            best_adv = adv
            current_objective = score
            best_score = evaluate_metrics(vae, images, adv)["decoded_diff_mse"]

    return best_adv, best_score


@torch.no_grad()
def visualize_attack(vae, original, adversarial, save_path, image_name="", attack_name=""):
    z_orig = encode_mode(vae, original)
    z_adv = encode_mode(vae, adversarial)
    dec_orig = decode_sample(vae, z_orig)
    dec_adv = decode_sample(vae, z_adv)

    img_orig = tensor_to_numpy_image(original)
    img_adv = tensor_to_numpy_image(adversarial)
    perturbation = np.clip(np.abs(img_adv - img_orig) * 10, 0, 1)

    lat_ch_orig = latent_to_rgb_channels(z_orig)
    lat_ch_adv = latent_to_rgb_channels(z_adv)
    if z_orig.dim() == 5:
        lat_diff_raw = (z_adv[0, :3, 0] - z_orig[0, :3, 0]).abs().cpu().float()
    else:
        lat_diff_raw = (z_adv[0, :3] - z_orig[0, :3]).abs().cpu().float()
    lat_diff = lat_diff_raw * 10
    for c in range(3):
        mn, mx = lat_diff[c].min(), lat_diff[c].max()
        lat_diff[c] = (lat_diff[c] - mn) / (mx - mn + 1e-8)
    lat_diff = lat_diff.permute(1, 2, 0).numpy()

    lat_pca_orig = latent_to_rgb_pca(z_orig)
    lat_pca_adv = latent_to_rgb_pca(z_adv)
    lat_pca_diff = np.clip(np.abs(lat_pca_adv - lat_pca_orig) * 10, 0, 1)

    dec_orig_np = tensor_to_numpy_image(dec_orig)
    dec_adv_np = tensor_to_numpy_image(dec_adv)
    dec_diff = np.clip(np.abs(dec_adv_np - dec_orig_np) * 10, 0, 1)

    metrics = evaluate_metrics(vae, original, adversarial)

    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.suptitle(
        f"{image_name} | best attack: {attack_name}\n"
        f"Pixel: MSE={metrics['pixel_mse']:.6f}, L∞={metrics['pixel_linf']:.4f} | "
        f"Latent: MSE={metrics['latent_mse']:.4f}, L∞={metrics['latent_linf']:.4f}\n"
        f"Recon MSE (orig)={metrics['recon_mse_orig']:.6f}, "
        f"Recon MSE (adv→orig)={metrics['recon_mse_adv_vs_orig']:.6f}, "
        f"Dec diff MSE={metrics['decoded_diff_mse']:.6f}",
        fontsize=12,
        y=0.98,
    )

    titles = [
        ["Original", "Adversarial", "Perturbation (×10)"],
        ["Latent (ch 0-2) Original", "Latent (ch 0-2) Adversarial", "Latent Ch Diff (×10)"],
        ["Latent (PCA) Original", "Latent (PCA) Adversarial", "Latent PCA Diff (×10)"],
        ["Decoded Original", "Decoded Adversarial", "Decoded Diff (×10)"],
    ]
    images_grid = [
        [img_orig, img_adv, perturbation],
        [lat_ch_orig, lat_ch_adv, lat_diff],
        [lat_pca_orig, lat_pca_adv, lat_pca_diff],
        [dec_orig_np, dec_adv_np, dec_diff],
    ]

    for row in range(4):
        for col in range(3):
            axes[row][col].imshow(images_grid[row][col])
            axes[row][col].set_title(titles[row][col], fontsize=11)
            axes[row][col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return metrics


def run_autoattack(
    vae,
    images: torch.Tensor,
    epsilon: float,
    apgd_steps: int,
    square_steps: int,
    apgd_restarts: int,
) -> tuple[torch.Tensor, str]:
    attack_plan = [
        ("apgd_recon", lambda: apgd_attack_vae(vae, images, epsilon, apgd_steps, "recon", apgd_restarts)),
        ("apgd_decoded", lambda: apgd_attack_vae(vae, images, epsilon, apgd_steps, "decoded", apgd_restarts)),
        ("apgd_latent", lambda: apgd_attack_vae(vae, images, epsilon, apgd_steps, "latent", apgd_restarts)),
        ("square_decoded", lambda: square_attack_vae(vae, images, epsilon, square_steps, "decoded")),
    ]

    best_adv = images.clone()
    best_name = "clean"
    best_score = evaluate_metrics(vae, images, best_adv)["decoded_diff_mse"]

    for attack_name, attack_fn in attack_plan:
        adv, score = attack_fn()
        if score > best_score:
            best_adv = adv
            best_name = attack_name
            best_score = score

    return best_adv, best_name


def collect_images(input_dir: str) -> list[Path]:
    input_path = Path(input_dir)
    return sorted(sum((list(input_path.glob(ext)) for ext in IMAGE_EXTS), []))


def load_vae(spec: ModelSpec, model_id: str, device: str):
    vae = spec.loader.from_pretrained(model_id, subfolder=spec.subfolder)
    return vae.to(device).float().eval()


def main():
    parser = argparse.ArgumentParser(description="AutoAttack-style ensemble attack for VAE robustness.")
    parser.add_argument("--model", type=str, default="sd15", choices=sorted(MODEL_SPECS))
    parser.add_argument("--input_dir", type=str, default="resources/test-images")
    parser.add_argument("--output_dir", type=str, default="results/autoattack")
    parser.add_argument("--model_id", type=str, default=None, help="Override the default HF model id")
    parser.add_argument("--epsilon", type=float, default=0.06)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--apgd_steps", type=int, default=100)
    parser.add_argument("--square_steps", type=int, default=200)
    parser.add_argument("--apgd_restarts", type=int, default=2)
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]
    model_id = args.model_id or spec.model_id
    image_size = args.image_size or spec.image_size

    if args.model in {"flux1", "flux2"}:
        assert image_size % 16 == 0, f"image_size must be divisible by 16, got {image_size}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {args.model} VAE from {model_id} ...")
    vae = load_vae(spec, model_id, device)
    print(f"VAE loaded. Latent channels: {vae.config.latent_channels}")

    image_files = collect_images(args.input_dir)
    print(f"Found {len(image_files)} images in {args.input_dir}")
    if not image_files:
        return

    output_path = Path(args.output_dir) / f"{args.model}_eps_{args.epsilon}"
    output_path.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        img_tensor = load_image(str(img_path), image_size, spec.is_video).to(device)
        adv_tensor, attack_name = run_autoattack(
            vae,
            img_tensor,
            epsilon=args.epsilon,
            apgd_steps=args.apgd_steps,
            square_steps=args.square_steps,
            apgd_restarts=args.apgd_restarts,
        )

        adv_pil = Image.fromarray((tensor_to_numpy_image(adv_tensor) * 255).astype(np.uint8))
        adv_pil.save(output_path / f"{img_path.stem}_adv.png")

        metrics = visualize_attack(
            vae,
            img_tensor,
            adv_tensor,
            save_path=output_path / f"{img_path.stem}_visualization.png",
            image_name=img_path.name,
            attack_name=attack_name,
        )
        metrics["image"] = img_path.name
        metrics["best_attack"] = attack_name
        all_metrics.append(metrics)

        print(f"  Best attack={attack_name}")
        print(f"  Pixel MSE={metrics['pixel_mse']:.6f}, L∞={metrics['pixel_linf']:.4f}")
        print(f"  Latent MSE={metrics['latent_mse']:.4f}, L∞={metrics['latent_linf']:.4f}")
        print(f"  Decoded diff MSE={metrics['decoded_diff_mse']:.6f}")

    scalar_keys = [k for k, v in all_metrics[0].items() if isinstance(v, (int, float))]
    summary = {
        "config": {
            **vars(args),
            "model_id": model_id,
            "image_size": image_size,
            "ensemble": ["apgd_recon", "apgd_decoded", "apgd_latent", "square_decoded"],
            "selection_metric": "decoded_diff_mse",
        },
        "per_image": all_metrics,
        "average": {
            key: round(float(np.mean([m[key] for m in all_metrics])), 6)
            for key in scalar_keys
        },
    }
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    main()
