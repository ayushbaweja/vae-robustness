#!/usr/bin/env python3
"""Plot attack-strength vs MSE for each VAE to compare robustness.

Curves show the per-image mean with a shaded band for variability across
images (mean ± stderr by default). Per-image data is read from each
experiment's summary.json so no re-running is required.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results")
OUT_DIR = "results/analysis"

# Trees of experiment results pooled into the per-image arrays. Each tree
# has the same internal layout (<tree>/<model_subdir>/eps_<eps>_<loss>/).
RESULT_TREES = [
    Path("results"),              # original 5-image set
    Path("results/imagenet25"),   # 25 ImageNet samples
]

# Map display name -> result subdir (matches analyze_results.py)
MODELS = {
    "SD 1.5":    "sd15_pgd",
    "FLUX.1":    "flux1_pgd",
    "FLUX.2":    "flux2_pgd",
    "CogVideoX": "cogvideox_pgd",
    "LTX Video": "ltx_pgd",
}

MODEL_COLORS = {
    "SD 1.5":    "#e41a1c",
    "FLUX.1":    "#377eb8",
    "FLUX.2":    "#4daf4a",
    "CogVideoX": "#984ea3",
    "LTX Video": "#ff7f00",
}

MODEL_ORDER = ["LTX Video", "CogVideoX", "FLUX.1", "FLUX.2", "SD 1.5"]

# "std"    — sample standard deviation across images
# "stderr" — std / sqrt(n), tighter band, representing uncertainty in the mean
BAND = "stderr"


def load_per_image():
    """Build (loss, model) -> sorted list of (eps, decoded_arr, latent_arr).

    Per-image metrics are pooled across every tree in RESULT_TREES so the
    bands reflect variability over all available images.
    """
    # First aggregate (loss, model, eps) -> (dec_list, lat_list)
    pooled = defaultdict(lambda: ([], []))
    for tree in RESULT_TREES:
        if not tree.exists():
            continue
        for model, subdir in MODELS.items():
            model_path = tree / subdir
            if not model_path.exists():
                continue
            for exp_dir in sorted(model_path.iterdir()):
                if not exp_dir.is_dir():
                    continue
                sp = exp_dir / "summary.json"
                if not sp.exists():
                    continue
                data = json.loads(sp.read_text())
                cfg = data["config"]
                eps = float(cfg["epsilon"])
                loss = cfg.get("loss", "pixel")
                per = data["per_image"]
                dec_list, lat_list = pooled[(loss, model, eps)]
                dec_list.extend(m["decoded_diff_mse"] for m in per)
                lat_list.extend(m["latent_mse"] for m in per)

    # Then collapse to (loss, model) -> [(eps, dec_arr, lat_arr), ...]
    groups = defaultdict(list)
    for (loss, model, eps), (dec_list, lat_list) in pooled.items():
        groups[(loss, model)].append(
            (eps, np.array(dec_list, dtype=float), np.array(lat_list, dtype=float))
        )
    for k in groups:
        groups[k].sort(key=lambda t: t[0])
    return groups


def _mean_band(arrs):
    """Per-epsilon mean and band edges across images."""
    means, los, his, ns = [], [], [], []
    for a in arrs:
        n = len(a)
        m = float(a.mean())
        if n > 1:
            s = float(a.std(ddof=1))
            spread = s / np.sqrt(n) if BAND == "stderr" else s
        else:
            spread = 0.0
        means.append(m)
        los.append(m - spread)
        his.append(m + spread)
        ns.append(n)
    return np.array(means), np.array(los), np.array(his), ns


def plot_metric(groups, metric_idx, ylabel, suptitle, filename, log_y=False):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    n_per_point = []
    for ax_idx, loss in enumerate(["pixel", "latent"]):
        ax = axes[ax_idx]
        for model in MODEL_ORDER:
            data = groups.get((loss, model), [])
            if not data:
                continue
            eps = np.array([d[0] for d in data])
            arrs = [d[metric_idx] for d in data]
            mean, lo, hi, ns = _mean_band(arrs)
            n_per_point.extend(ns)

            color = MODEL_COLORS[model]
            ax.fill_between(eps, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(eps, mean, "o-", color=color, label=model,
                    linewidth=2.5, markersize=7)

        ax.set_xlabel("Attack Strength (ε, L∞ budget)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f"PGD Attack — {loss} loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0.02, 0.04, 0.06, 0.1, 0.15, 0.2])
        if log_y:
            ax.set_yscale("log")

    band_label = "±1 SEM" if BAND == "stderr" else "±1 SD"
    if n_per_point:
        lo_n, hi_n = min(n_per_point), max(n_per_point)
        n_text = f"n={lo_n}" if lo_n == hi_n else f"n={lo_n}–{hi_n}"
    else:
        n_text = "n=?"
    fig.suptitle(f"{suptitle}  (shaded: {band_label} across images, {n_text})",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = f"{OUT_DIR}/{filename}"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    groups = load_per_image()
    if not groups:
        print(f"No experiment results found under {RESULTS_DIR}/")
        return

    # metric_idx: 1 = decoded_diff_mse arrays, 2 = latent_mse arrays
    plot_metric(groups, 1,
                "Decoded Diff MSE (output damage)",
                "VAE Robustness: Attack Strength vs Output Damage",
                "robustness_decoded_mse.png")

    plot_metric(groups, 2,
                "Latent MSE",
                "VAE Robustness: Attack Strength vs Latent Displacement",
                "robustness_latent_mse.png", log_y=True)


if __name__ == "__main__":
    main()
