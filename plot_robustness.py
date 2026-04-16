#!/usr/bin/env python3
"""Plot attack-strength vs MSE for each VAE to compare robustness."""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

CSV_PATH = "results/analysis/sweep_table.csv"
OUT_DIR = "results/analysis"

MODEL_COLORS = {
    "SD 1.5":    "#e41a1c",
    "FLUX.1":    "#377eb8",
    "FLUX.2":    "#4daf4a",
    "CogVideoX": "#984ea3",
    "LTX Video": "#ff7f00",
}

MODEL_ORDER = ["LTX Video", "CogVideoX", "FLUX.1", "FLUX.2", "SD 1.5"]

# Load CSV
rows = []
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        r["epsilon"] = float(r["epsilon"])
        r["decoded_diff_mse"] = float(r["decoded_diff_mse"])
        r["latent_mse"] = float(r["latent_mse"])
        rows.append(r)

# Group: (loss_mode, model) -> sorted list of (eps, mse)
groups = defaultdict(list)
for r in rows:
    groups[(r["loss_mode"], r["model"])].append((r["epsilon"], r["decoded_diff_mse"], r["latent_mse"]))
for k in groups:
    groups[k].sort()

# --- Plot 1: Decoded Diff MSE (output-space damage) ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

for ax_idx, loss in enumerate(["pixel", "latent"]):
    ax = axes[ax_idx]
    for model in MODEL_ORDER:
        data = groups.get((loss, model), [])
        if not data:
            continue
        eps = [d[0] for d in data]
        mse = [d[1] for d in data]
        ax.plot(eps, mse, "o-", color=MODEL_COLORS[model], label=model,
                linewidth=2.5, markersize=7)

    ax.set_xlabel("Attack Strength (ε, L∞ budget)", fontsize=13)
    ax.set_ylabel("Decoded Diff MSE (output damage)", fontsize=13)
    ax.set_title(f"PGD Attack — {loss} loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.02, 0.04, 0.06, 0.1, 0.15, 0.2])

fig.suptitle("VAE Robustness: Attack Strength vs Output Damage",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
out = f"{OUT_DIR}/robustness_decoded_mse.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# --- Plot 2: Latent MSE (latent-space displacement) ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

for ax_idx, loss in enumerate(["pixel", "latent"]):
    ax = axes[ax_idx]
    for model in MODEL_ORDER:
        data = groups.get((loss, model), [])
        if not data:
            continue
        eps = [d[0] for d in data]
        lat_mse = [d[2] for d in data]
        ax.plot(eps, lat_mse, "o-", color=MODEL_COLORS[model], label=model,
                linewidth=2.5, markersize=7)

    ax.set_xlabel("Attack Strength (ε, L∞ budget)", fontsize=13)
    ax.set_ylabel("Latent MSE", fontsize=13)
    ax.set_title(f"PGD Attack — {loss} loss", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.02, 0.04, 0.06, 0.1, 0.15, 0.2])

fig.suptitle("VAE Robustness: Attack Strength vs Latent Displacement",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
out = f"{OUT_DIR}/robustness_latent_mse.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
