#!/usr/bin/env python3
"""Regenerate the summary figures from aggregate numbers only.

No imaging data is read. Every value below is a cohort mean copied from the
result tables described in docs/08-results.md.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

C_DL = "#2b6cb0"
C_ATLAS = "#a0aec0"
C_BAD = "#c53030"
C_OK = "#2f855a"


def hammers_three_way() -> None:
    methods = ["nnU-Net\n(Dataset001, 5-fold)", "MAPER\n(leave-one-out)", "ANTs JLF\n(leave-one-out)"]
    dice_l = [0.708, 0.659, 0.659]
    dice_r = [0.693, 0.627, 0.613]
    jac_l = [0.551, 0.495, 0.494]
    jac_r = [0.534, 0.462, 0.445]
    colors = [C_DL, C_ATLAS, C_ATLAS]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(methods))
    w = 0.38
    for ax, left, right, name in (
        (axes[0], dice_l, dice_r, "Dice"),
        (axes[1], jac_l, jac_r, "Jaccard"),
    ):
        b1 = ax.bar(x - w / 2, left, w, color=colors, label="Left PC")
        b2 = ax.bar(x + w / 2, right, w, color=colors, alpha=0.6, hatch="//", label="Right PC")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}",
                        ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, methods)
        ax.set_ylim(0, 0.85)
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs manual labels (Hammers n=30)")
    axes[1].axhline(0.50, ls="--", c="k", lw=1)
    axes[1].text(-0.55, 0.72, "dashed line: published MAPER\nbar, Jaccard ≈ 0.50\n(Steinbart et al. 2023)",
                 fontsize=8, va="bottom", ha="left")
    axes[0].legend(handles=[b1[0], b2[0]], labels=["Left", "Right"], frameon=False, loc="upper right")
    fig.suptitle("Stage A: supervised nnU-Net beats both atlas methods in-domain", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "hammers_three_way.png", dpi=160)
    plt.close(fig)


def site_gap() -> None:
    labels = [
        "Dataset003\nHammers+ADNI\n(Hammers val)",
        "Dataset003\nHammers+ADNI\n(ADNI val)",
        "Dataset004\nharmonized\n(ADNI val)",
        "Dataset005\nADNI-only\n(ADNI val)",
    ]
    vals = [0.68, 0.02, 0.04, 0.017]
    colors = [C_OK, C_BAD, C_BAD, C_BAD]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, vals, color=colors)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}",
                ha="center", va="bottom")
    ax.axhline(0.40, ls="--", c="k", lw=1)
    ax.text(3.45, 0.41, "pass bar 0.40 (fixed before results)", ha="right", fontsize=9)
    ax.axhline(0.10, ls=":", c=C_BAD, lw=1)
    ax.text(3.45, 0.11, "fail bar 0.10", ha="right", fontsize=9, color=C_BAD)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Mean Dice (left PC), mid-training check")
    ax.set_title("Stage B: the same recipe learned Hammers but never learned ADNI", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "site_gap.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    hammers_three_way()
    site_gap()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
