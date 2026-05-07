#!/usr/bin/env python3
"""Focused AutoAttack-vs-PGD visualization for SD 1.5 and FLUX.1."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "analysis_autoattack"
EPSILONS = [0.02, 0.04, 0.06, 0.1, 0.15, 0.2]

MODELS = {
    "SD 1.5": ("sd15", "sd15_pgd", "#e41a1c"),
    "FLUX.1": ("flux1", "flux1_pgd", "#377eb8"),
}


def load_summary(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_model_series(auto_key: str, pgd_key: str):
    rows = []
    for eps in EPSILONS:
        auto_path = RESULTS_DIR / "autoattack" / f"{auto_key}_eps_{eps}" / "summary.json"
        pgd_path = RESULTS_DIR / pgd_key / f"eps_{eps}_pixel" / "summary.json"
        auto = load_summary(auto_path)
        pgd = load_summary(pgd_path)
        if auto is None or pgd is None:
            continue
        rows.append({
            "eps": eps,
            "auto": auto["average"]["decoded_diff_mse"],
            "pgd": pgd["average"]["decoded_diff_mse"],
        })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)

    for ax, (label, (auto_key, pgd_key, color)) in zip(axes, MODELS.items()):
        rows = load_model_series(auto_key, pgd_key)
        if not rows:
            ax.set_title(f"{label}\n(no data)")
            ax.axis("off")
            continue

        xs = np.array([r["eps"] for r in rows], dtype=float)
        auto = np.array([r["auto"] for r in rows], dtype=float)
        pgd = np.array([r["pgd"] for r in rows], dtype=float)
        gain = auto / pgd

        ax.plot(xs, pgd, "o--", color=color, alpha=0.55, linewidth=2.2, markersize=6, label="PGD pixel-loss")
        ax.plot(xs, auto, "o-", color=color, linewidth=2.8, markersize=7, label="AutoAttack")
        ax.fill_between(xs, pgd, auto, color=color, alpha=0.14)

        for x, y_auto, g in zip(xs, auto, gain):
            ax.text(x, y_auto, f"{g:.1f}x", fontsize=8, ha="center", va="bottom")

        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epsilon (L∞ budget)")
        ax.set_xticks(EPSILONS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel("Decoded Diff MSE")
    fig.suptitle("AutoAttack vs PGD Pixel-Loss: SD 1.5 and FLUX.1", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = OUT_DIR / "autoattack_vs_pgd_sd15_flux1.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
