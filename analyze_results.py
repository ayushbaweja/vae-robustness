#!/usr/bin/env python3
"""Analyze and plot results from the VAE robustness sweep.

Reads all summary.json files, generates:
  - results/analysis/sweep_table.csv           — full results table
  - results/analysis/sweep_decoded_damage.png  — decoded diff MSE vs epsilon
  - results/analysis/sweep_latent_mse.png      — latent MSE vs epsilon
  - results/analysis/sweep_amplification.png   — amplification ratio vs epsilon
  - results/analysis/matrix_heatmap.png        — heatmap of decoded damage
"""

import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

RESULTS_DIR = Path("results")
ANALYSIS_DIR = RESULTS_DIR / "analysis"

# Model display names and result directories
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

EPSILONS = [0.02, 0.04, 0.06, 0.1, 0.15, 0.2]
LOSSES = ["pixel", "latent"]


def load_all_results():
    """Load all summary.json files into a structured dict."""
    results = {}  # (model_name, epsilon, loss_mode) -> average metrics

    for model_name, result_dir in MODELS.items():
        model_path = RESULTS_DIR / result_dir
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
            eps = config["epsilon"]
            loss = config.get("loss", "pixel")  # legacy runs default to pixel

            key = (model_name, eps, loss)
            results[key] = {
                "config": config,
                "average": data["average"],
                "per_image": data["per_image"],
            }

    return results


def write_csv(results):
    """Write full results table as CSV."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS_DIR / "sweep_table.csv"

    rows = []
    for (model, eps, loss), data in sorted(results.items()):
        avg = data["average"]
        rows.append({
            "model": model,
            "epsilon": eps,
            "loss_mode": loss,
            "pixel_mse": avg["pixel_mse"],
            "pixel_linf": avg["pixel_linf"],
            "latent_mse": avg["latent_mse"],
            "latent_linf": avg["latent_linf"],
            "recon_mse_orig": avg["recon_mse_orig"],
            "recon_mse_adv_vs_orig": avg["recon_mse_adv_vs_orig"],
            "decoded_diff_mse": avg["decoded_diff_mse"],
            "amplification": avg["latent_mse"] / avg["pixel_mse"] if avg["pixel_mse"] > 0 else 0,
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path} ({len(rows)} rows)")
    return rows


def plot_sweep(results, metric, ylabel, title, filename, log_y=False):
    """Plot metric vs epsilon, one subplot per loss mode, one curve per model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax_idx, loss in enumerate(LOSSES):
        ax = axes[ax_idx]
        for model_name in MODELS:
            xs, ys = [], []
            for eps in EPSILONS:
                key = (model_name, eps, loss)
                if key in results:
                    xs.append(eps)
                    avg = results[key]["average"]
                    if metric == "amplification":
                        val = avg["latent_mse"] / avg["pixel_mse"] if avg["pixel_mse"] > 0 else 0
                    else:
                        val = avg[metric]
                    ys.append(val)

            if xs:
                ax.plot(xs, ys, "o-", color=MODEL_COLORS[model_name],
                        label=model_name, linewidth=2, markersize=6)

        ax.set_xlabel("Epsilon (L∞ budget)", fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Loss: {loss}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(EPSILONS)
        if log_y:
            ax.set_yscale("log")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = ANALYSIS_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def plot_heatmap(results):
    """Plot heatmap of decoded_diff_mse for each (model, epsilon) pair."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, loss in enumerate(LOSSES):
        ax = axes[ax_idx]
        model_names = list(MODELS.keys())
        data = np.full((len(model_names), len(EPSILONS)), np.nan)

        for i, model in enumerate(model_names):
            for j, eps in enumerate(EPSILONS):
                key = (model, eps, loss)
                if key in results:
                    data[i, j] = results[key]["average"]["decoded_diff_mse"]

        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(EPSILONS)))
        ax.set_xticklabels([str(e) for e in EPSILONS])
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_xlabel("Epsilon")
        ax.set_title(f"Decoded Diff MSE — {loss} loss", fontsize=12)

        # Annotate cells
        for i in range(len(model_names)):
            for j in range(len(EPSILONS)):
                val = data[i, j]
                if not np.isnan(val):
                    color = "white" if val > np.nanmax(data) * 0.6 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Decoded Damage Heatmap (MSE)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = ANALYSIS_DIR / "matrix_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def plot_pixel_vs_latent(results):
    """For each model at each epsilon, compare pixel vs latent loss decoded damage."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in MODELS:
        pixel_vals, latent_vals, eps_vals = [], [], []
        for eps in EPSILONS:
            pk = (model_name, eps, "pixel")
            lk = (model_name, eps, "latent")
            if pk in results and lk in results:
                pixel_vals.append(results[pk]["average"]["decoded_diff_mse"])
                latent_vals.append(results[lk]["average"]["decoded_diff_mse"])
                eps_vals.append(eps)

        if pixel_vals:
            color = MODEL_COLORS[model_name]
            ax.plot(eps_vals, pixel_vals, "o-", color=color, linewidth=2,
                    markersize=6, label=f"{model_name} (pixel)")
            ax.plot(eps_vals, latent_vals, "s--", color=color, linewidth=2,
                    markersize=6, alpha=0.6, label=f"{model_name} (latent)")

    ax.set_xlabel("Epsilon (L∞ budget)", fontsize=12)
    ax.set_ylabel("Decoded Diff MSE", fontsize=12)
    ax.set_title("Pixel vs Latent Loss: Decoded Damage Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(EPSILONS)

    out_path = ANALYSIS_DIR / "pixel_vs_latent_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def print_summary_table(results):
    """Print a formatted summary to stdout."""
    print("\n" + "=" * 90)
    print("SUMMARY: Decoded Diff MSE across all experiments")
    print("=" * 90)

    for loss in LOSSES:
        print(f"\n{'─' * 90}")
        print(f"  Loss mode: {loss}")
        print(f"{'─' * 90}")
        header = f"{'Model':<12}" + "".join(f"{'ε=' + str(e):>12}" for e in EPSILONS)
        print(header)
        print("─" * len(header))

        for model in MODELS:
            row = f"{model:<12}"
            for eps in EPSILONS:
                key = (model, eps, loss)
                if key in results:
                    val = results[key]["average"]["decoded_diff_mse"]
                    row += f"{val:>12.4f}"
                else:
                    row += f"{'—':>12}"
            print(row)

    print()


def main():
    results = load_all_results()
    print(f"Loaded {len(results)} experiment results")

    if not results:
        print("No results found. Run experiments first with run_sweep.sh")
        return

    print_summary_table(results)
    rows = write_csv(results)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    plot_sweep(results, "decoded_diff_mse",
               "Decoded Diff MSE", "Decoded Damage vs Perturbation Budget",
               "sweep_decoded_damage.png")

    plot_sweep(results, "latent_mse",
               "Latent MSE", "Latent Displacement vs Perturbation Budget",
               "sweep_latent_mse.png", log_y=True)

    plot_sweep(results, "amplification",
               "Latent MSE / Pixel MSE", "Latent Amplification Factor vs Perturbation Budget",
               "sweep_amplification.png", log_y=True)

    plot_heatmap(results)
    plot_pixel_vs_latent(results)

    print(f"\nAll analysis outputs saved to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
