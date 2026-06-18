"""Shared helpers for the encoder/decoder Jacobian experiments.

These reuse the exact load/round-trip convention of ``pgd_flux_vae.py`` /
``pgd_flux2_vae.py``:

* images in ``[-1, 1]``, shape ``(1, 3, 512, 512)``;
* ``z = vae.encode(x).latent_dist.mode()`` (deterministic mean);
* ``recon = vae.decode(z).sample``;
* **no** ``scaling_factor`` / ``shift_factor`` and **no** FLUX.2 pipeline
  BatchNorm / 2x2 packing — none of those are in the attacked path.

All VAEs run on CUDA in fp32 (``.float()``) for stable gradients/Jacobians.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, AutoencoderKLFlux2


# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "flux1": {
        "cls": AutoencoderKL,
        "id": "black-forest-labs/FLUX.1-dev",
        "label": "FLUX.1-dev",
    },
    "flux2": {
        "cls": AutoencoderKLFlux2,
        "id": "black-forest-labs/FLUX.2-dev",
        "label": "FLUX.2-dev",
    },
}

MODEL_ORDER = ["flux1", "flux2"]


def load_vae(key: str, device: str = "cuda"):
    """Load a VAE exactly as the PGD scripts do: ``subfolder='vae'``, fp32, eval."""
    spec = MODELS[key]
    vae = spec["cls"].from_pretrained(spec["id"], subfolder="vae")
    vae = vae.to(device).float().eval()
    return vae


def model_metadata(key: str, vae) -> dict:
    spec = MODELS[key]
    return {
        "key": key,
        "label": spec["label"],
        "model_id": spec["id"],
        "vae_class": spec["cls"].__name__,
        "latent_channels": int(vae.config.latent_channels),
    }


# ── Image IO (identical convention to pgd_flux_vae.py) ─────────────────────────

def load_image(path: str, size: int = 512) -> torch.Tensor:
    """Load image, resize, return tensor in [-1, 1] with shape (1, 3, H, W)."""
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) or (3,H,W) in [-1,1] -> (H,W,3) in [0,1]."""
    if t.dim() == 4:
        t = t[0]
    return (t.permute(1, 2, 0).cpu().float() * 0.5 + 0.5).clamp(0, 1).numpy()


def list_images(input_dir: str, limit: int | None = None):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG")
    path = Path(input_dir)
    files = sorted(set(sum([list(path.glob(e)) for e in exts], [])))
    if limit is not None:
        files = files[:limit]
    return files


# ── Round-trip primitives ─────────────────────────────────────────────────────

def encode(vae, x: torch.Tensor) -> torch.Tensor:
    """Deterministic latent mean, as in the attacked path."""
    return vae.encode(x).latent_dist.mode()


def decode(vae, z: torch.Tensor) -> torch.Tensor:
    return vae.decode(z).sample


def channel_std(z: torch.Tensor) -> torch.Tensor:
    """Per-channel std over spatial dims, returned broadcastable as (1, C, 1, 1).

    This is the fairness normalization: a "unit" latent perturbation means the
    same thing for the 16-channel and 32-channel latent spaces only after each
    channel is divided by its own std. Reads C from the tensor (never hardcoded).
    """
    # z: (1, C, h, w) -> std over (batch, h, w) per channel
    s = z.std(dim=(0, 2, 3))  # (C,)
    return s.view(1, -1, 1, 1)


# ── PGD attack (same recipe as pgd_flux_vae.py) ───────────────────────────────

def pgd_attack_vae(
    vae,
    images: torch.Tensor,
    epsilon: float = 0.06,
    alpha: float = 0.01,
    num_iter: int = 40,
    loss_mode: str = "pixel",
) -> torch.Tensor:
    """PGD attack on the VAE under an L-inf budget.

    loss_mode:
        "pixel"  -- max ||decode(encode(x+d)) - x||^2  (gradient through full VAE)
        "latent" -- max ||encode(x+d) - encode(x)||^2  (gradient through encoder)
    """
    vae.eval()

    with torch.no_grad():
        z_orig = encode(vae, images)

    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, -1, 1) - images
    delta.requires_grad = True

    for _ in range(num_iter):
        adv = images + delta
        z = encode(vae, adv)
        if loss_mode == "latent":
            loss = F.mse_loss(z, z_orig, reduction="sum")
        else:
            recon = decode(vae, z)
            loss = F.mse_loss(recon, images, reduction="sum")

        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(images + delta, -1, 1) - images
        delta.requires_grad = True

    return (images + delta).detach()


# ── Small numeric helpers ─────────────────────────────────────────────────────

def l2(t: torch.Tensor) -> float:
    return float(t.flatten().norm().item())


def mean_std(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()) if a.size else float("nan"),
        "std": float(a.std()) if a.size else float("nan"),
        "n": int(a.size),
    }


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")
