#!/usr/bin/env python3
"""Analyze and plot results from the AutoAttack-style VAE sweep."""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
AUTOATTACK_DIR = RESULTS_DIR / "autoattack"
ANALYSIS_DIR = RESULTS_DIR / "analysis_autoattack"

MODELS = {
    "SD 1.5": "sd15",
    "FLUX.1": "flux1",
    "FLUX.2": "flux2",
    "CogVideoX": "cogvideox",
    "LTX Video": "ltx",
}

PGD_PIXEL_DIRS = {
    "SD 1.5": "sd15_pgd",
    "FLUX.1": "flux1_pgd",
    "FLUX.2": "flux2_pgd",
    "CogVideoX": "cogvideox_pgd",
    "LTX Video": "ltx_pgd",
}

MODEL_COLORS = {
    "SD 1.5": "#e41a1c",
    "FLUX.1": "#377eb8",
    "FLUX.2": "#4daf4a",
    "CogVideoX": "#984ea3",
    "LTX Video": "#ff7f00",
}

ATTACK_COLORS = {
    "apgd_recon": "#1b9e77",
    "apgd_decoded": "#d95f02",
    "apgd_latent": "#7570b3",
    "square_decoded": "#e7298a",
    "clean": "#666666",
}

EPSILONS = [0.02, 0.04, 0.06, 0.1, 0.15, 0.2]


def load_autoattack_results():
    results = {}
    for model_name, model_key in MODELS.items():
        for eps in EPSILONS:
            summary_path = AUTOATTACK_DIR / f"{model_key}_eps_{eps}" / "summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path) as f:
                data = json.load(f)
            results[(model_name, eps)] = {
                "config": data["config"],
                "average": data["average"],
                "per_image": data["per_image"],
            }
    return results


def load_pgd_pixel_results():
    results = {}
    for model_name, subdir in PGD_PIXEL_DIRS.items():
        model_path = RESULTS_DIR / subdir
        if not model_path.exists():
            continue
        for exp_dir in sorted(model_path.iterdir()):
            if not exp_dir.is_dir():
                continue
            summary_path = exp_dir / "summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path) as f:
                data = json.load(f)
            config = data["config"]
            if config.get("loss", "pixel") != "pixel":
                continue
            results[(model_name, float(config["epsilon"]))] = {
                "average": data["average"],
                "per_image": data["per_image"],
            }
    return results


def write_csv(results):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS_DIR / "autoattack_sweep_table.csv"

    rows = []
    for (model, eps), data in sorted(results.items()):
        avg = data["average"]
        counts = Counter(m.get("best_attack", "clean") for m in data["per_image"])
        rows.append({
            "model": model,
            "epsilon": eps,
            "pixel_mse": avg["pixel_mse"],
            "pixel_linf": avg["pixel_linf"],
            "latent_mse": avg["latent_mse"],
            "latent_linf": avg["latent_linf"],
            "recon_mse_orig": avg["recon_mse_orig"],
            "recon_mse_adv_vs_orig": avg["recon_mse_adv_vs_orig"],
            "decoded_diff_mse": avg["decoded_diff_mse"],
            "amplification": avg["latent_mse"] / avg["pixel_mse"] if avg["pixel_mse"] > 0 else 0.0,
            "apgd_recon_wins": counts["apgd_recon"],
            "apgd_decoded_wins": counts["apgd_decoded"],
            "apgd_latent_wins": counts["apgd_latent"],
            "square_decoded_wins": counts["square_decoded"],
            "clean_wins": counts["clean"],
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path} ({len(rows)} rows)")


def _per_image_values(per_image, metric):
    if metric == "amplification":
        vals = []
        for metrics in per_image:
            pixel_mse = metrics["pixel_mse"]
            vals.append(metrics["latent_mse"] / pixel_mse if pixel_mse > 0 else 0.0)
        return np.array(vals, dtype=float)
    return np.array([metrics[metric] for metrics in per_image], dtype=float)


def plot_metric(results, metric, ylabel, title, filename, log_y=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    n_seen = []

    for model_name in MODELS:
        xs, means, los, his = [], [], [], []
        for eps in EPSILONS:
            key = (model_name, eps)
            if key not in results:
                continue
            vals = _per_image_values(results[key]["per_image"], metric)
            if len(vals) == 0:
                continue
            n = len(vals)
            mean = float(vals.mean())
            spread = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            xs.append(eps)
            means.append(mean)
            los.append(mean - spread)
            his.append(mean + spread)
            n_seen.append(n)

        if xs:
            color = MODEL_COLORS[model_name]
            ax.fill_between(xs, los, his, color=color, alpha=0.18, linewidth=0)
            ax.plot(xs, means, "o-", color=color, label=model_name, linewidth=2.5, markersize=7)

    ax.set_xlabel("Epsilon (L∞ budget)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks(EPSILONS)
    if log_y:
        ax.set_yscale("log")

    if n_seen:
        lo_n, hi_n = min(n_seen), max(n_seen)
        n_text = f"n={lo_n}" if lo_n == hi_n else f"n={lo_n}-{hi_n}"
        fig.suptitle(f"Shaded band: ±1 SEM across images ({n_text})", fontsize=11, y=0.98)

    plt.tight_layout()
    out_path = ANALYSIS_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def plot_heatmap(results):
    model_names = list(MODELS.keys())
    data = np.full((len(model_names), len(EPSILONS)), np.nan)

    for row, model in enumerate(model_names):
        for col, eps in enumerate(EPSILONS):
            key = (model, eps)
            if key in results:
                data[row, col] = results[key]["average"]["decoded_diff_mse"]

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(EPSILONS)))
    ax.set_xticklabels([str(eps) for eps in EPSILONS])
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Epsilon")
    ax.set_title("AutoAttack Decoded Damage Heatmap", fontsize=14, fontweight="bold")

    max_val = np.nanmax(data) if not np.isnan(data).all() else 0.0
    for row in range(len(model_names)):
        for col in range(len(EPSILONS)):
            val = data[row, col]
            if not np.isnan(val):
                color = "white" if max_val and val > max_val * 0.6 else "black"
                ax.text(col, row, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    out_path = ANALYSIS_DIR / "autoattack_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def plot_best_attack_share(results):
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(10, 12), sharex=True)
    if len(MODELS) == 1:
        axes = [axes]

    attack_names = [name for name in ATTACK_COLORS if name != "clean"]

    for ax, model_name in zip(axes, MODELS):
        bottoms = np.zeros(len(EPSILONS), dtype=float)
        for attack_name in attack_names:
            shares = []
            for eps in EPSILONS:
                key = (model_name, eps)
                if key not in results:
                    shares.append(0.0)
                    continue
                per_image = results[key]["per_image"]
                counts = Counter(m.get("best_attack", "clean") for m in per_image)
                shares.append(counts[attack_name] / max(len(per_image), 1))
            ax.bar(
                EPSILONS,
                shares,
                bottom=bottoms,
                width=0.015,
                color=ATTACK_COLORS[attack_name],
                label=attack_name if model_name == list(MODELS.keys())[0] else None,
            )
            bottoms += np.array(shares)

        ax.set_ylabel(model_name, rotation=0, labelpad=42, va="center")
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].legend(fontsize=9, ncol=4, loc="upper center")
    axes[-1].set_xlabel("Epsilon")
    axes[-1].set_xticks(EPSILONS)
    fig.suptitle("Best-Attack Win Rate per Image", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = ANALYSIS_DIR / "autoattack_best_attack_share.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def plot_vs_pgd_pixel(autoattack_results, pgd_results):
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in MODELS:
        xs, auto_vals, pgd_vals = [], [], []
        for eps in EPSILONS:
            auto_key = (model_name, eps)
            pgd_key = (model_name, eps)
            if auto_key not in autoattack_results or pgd_key not in pgd_results:
                continue
            xs.append(eps)
            auto_vals.append(autoattack_results[auto_key]["average"]["decoded_diff_mse"])
            pgd_vals.append(pgd_results[pgd_key]["average"]["decoded_diff_mse"])

        if xs:
            color = MODEL_COLORS[model_name]
            ax.plot(xs, pgd_vals, "o--", color=color, alpha=0.55, linewidth=2, markersize=6, label=f"{model_name} PGD-pixel")
            ax.plot(xs, auto_vals, "o-", color=color, linewidth=2.5, markersize=7, label=f"{model_name} AutoAttack")

    ax.set_xlabel("Epsilon (L∞ budget)", fontsize=12)
    ax.set_ylabel("Decoded Diff MSE", fontsize=12)
    ax.set_title("AutoAttack vs PGD Pixel-Loss", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xticks(EPSILONS)

    plt.tight_layout()
    out_path = ANALYSIS_DIR / "autoattack_vs_pgd_pixel.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def plot_vs_pgd_gain_heatmap(autoattack_results, pgd_results):
    model_names = list(MODELS.keys())
    ratio_data = np.full((len(model_names), len(EPSILONS)), np.nan)
    delta_data = np.full((len(model_names), len(EPSILONS)), np.nan)

    for row, model_name in enumerate(model_names):
        for col, eps in enumerate(EPSILONS):
            key = (model_name, eps)
            if key not in autoattack_results or key not in pgd_results:
                continue

            auto_val = autoattack_results[key]["average"]["decoded_diff_mse"]
            pgd_val = pgd_results[key]["average"]["decoded_diff_mse"]
            ratio_data[row, col] = auto_val / pgd_val if pgd_val > 0 else np.nan
            delta_data[row, col] = auto_val - pgd_val

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    plots = [
        (
            axes[0],
            ratio_data,
            "AutoAttack / PGD",
            "AutoAttack-to-PGD Damage Ratio",
            "viridis",
        ),
        (
            axes[1],
            delta_data,
            "Decoded Diff MSE",
            "AutoAttack Minus PGD Damage",
            "magma",
        ),
    ]

    for ax, data, cbar_label, title, cmap in plots:
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(EPSILONS)))
        ax.set_xticklabels([str(eps) for eps in EPSILONS])
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_xlabel("Epsilon")
        ax.set_title(title, fontsize=13, fontweight="bold")

        finite_vals = data[np.isfinite(data)]
        max_val = float(finite_vals.max()) if finite_vals.size else 0.0
        for row in range(len(model_names)):
            for col in range(len(EPSILONS)):
                val = data[row, col]
                if np.isnan(val):
                    continue
                color = "white" if max_val and val > max_val * 0.6 else "black"
                ax.text(col, row, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(cbar_label)

    fig.suptitle("How Much Stronger AutoAttack Is Than PGD Pixel-Loss", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = ANALYSIS_DIR / "autoattack_vs_pgd_gain_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def print_summary_table(results):
    print("\n" + "=" * 86)
    print("AUTOATTACK SUMMARY: Decoded Diff MSE")
    print("=" * 86)
    header = f"{'Model':<12}" + "".join(f"{'ε=' + str(eps):>12}" for eps in EPSILONS)
    print(header)
    print("-" * len(header))

    for model_name in MODELS:
        row = f"{model_name:<12}"
        for eps in EPSILONS:
            key = (model_name, eps)
            if key in results:
                val = results[key]["average"]["decoded_diff_mse"]
                row += f"{val:>12.4f}"
            else:
                row += f"{'—':>12}"
        print(row)
    print()


def main():
    autoattack_results = load_autoattack_results()
    print(f"Loaded {len(autoattack_results)} autoattack results")

    if not autoattack_results:
        print(f"No results found under {AUTOATTACK_DIR}/")
        print("Run experiments first with bash run_autoattack_sweep.sh")
        return

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print_summary_table(autoattack_results)
    write_csv(autoattack_results)

    plot_metric(
        autoattack_results,
        "decoded_diff_mse",
        "Decoded Diff MSE",
        "AutoAttack: Output Damage vs Perturbation Budget",
        "autoattack_decoded_damage.png",
    )
    plot_metric(
        autoattack_results,
        "latent_mse",
        "Latent MSE",
        "AutoAttack: Latent Displacement vs Perturbation Budget",
        "autoattack_latent_mse.png",
        log_y=True,
    )
    plot_metric(
        autoattack_results,
        "amplification",
        "Latent MSE / Pixel MSE",
        "AutoAttack: Latent Amplification vs Perturbation Budget",
        "autoattack_amplification.png",
        log_y=True,
    )
    plot_heatmap(autoattack_results)
    plot_best_attack_share(autoattack_results)

    pgd_pixel_results = load_pgd_pixel_results()
    if pgd_pixel_results:
        plot_vs_pgd_pixel(autoattack_results, pgd_pixel_results)
        plot_vs_pgd_gain_heatmap(autoattack_results, pgd_pixel_results)

    print(f"\nAll analysis outputs saved to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
