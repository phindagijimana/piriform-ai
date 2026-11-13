#!/usr/bin/env python3
"""Regenerate the summary figures from aggregate numbers only.

No imaging data is read. Bar-chart values are cohort means from the result
tables in docs/results.rst. Training curves are read from
data/training_curves.json, written by extract_training_curves.py.
"""
import json
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
    fig.suptitle("Stage A: nnU-Net has higher overlap than both atlas methods in-domain", fontweight="bold")
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


def _ema(x, a=0.9):
    out, m = [], None
    for v in x:
        if v is None:
            out.append(m)
            continue
        m = v if m is None else a * m + (1 - a) * v
        out.append(m)
    return out


def _curves():
    return json.loads((OUT / "data" / "training_curves.json").read_text())


def training_nnunet() -> None:
    folds = _curves()["nnunet"]["001 Hammers"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (name, f) in enumerate(sorted(folds.items())):
        lab = "train" if i == 0 else None
        ax1.plot(f["epoch"], f["train_loss"], c=C_DL, lw=0.6, alpha=0.6, label=lab)
        ax1.plot(f["epoch"], f["val_loss"], c=C_BAD, lw=0.6, alpha=0.6, label="validation" if i == 0 else None)
        ax2.plot(f["epoch"], _ema(f["pseudo_dice"]), lw=1.2, label=name.replace("_", " "))
    ax1.set_xlabel("epoch (250 iterations each)")
    ax1.set_ylabel("Dice + CE loss (lower is better)")
    ax1.set_title("Loss, 5 folds", fontweight="bold")
    ax1.legend(frameon=False)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation pseudo-Dice (EMA 0.9)")
    ax2.set_ylim(0, 0.85)
    ax2.set_title("Validation overlap, 5 folds", fontweight="bold")
    lr = sorted(folds.items())[0][1]
    axr = ax2.twinx()
    axr.plot(lr["epoch"], lr["lr"], c="k", ls=":", lw=1)
    axr.set_ylabel("learning rate (poly 0.9)")
    axr.spines["right"].set_visible(True)
    ax2.legend(frameon=False, loc="center right", fontsize=9, ncol=2)
    fig.suptitle("Stage A (Dataset001): nnU-Net training on 24 Hammers brains per fold", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "training_nnunet.png", dpi=160)
    plt.close(fig)


def training_sites() -> None:
    nn = _curves()["nnunet"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    style = {
        "001 Hammers": C_DL, "002 Hammers crop": "#63b3ed", "003 Hammers+ADNI": "#dd6b20",
        "004 Hammers+ADNI harmonised": "#b7791f", "005 ADNI only": C_BAD,
    }
    for name, col in style.items():
        f = max(nn[name].values(), key=lambda v: len(v["epoch"]))
        ax1.plot(f["epoch"], _ema(f["pseudo_dice"]), c=col, lw=1.3, label=name)
        ax2.plot(f["epoch"], f["train_loss"], c=col, lw=0.8, label=name)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation pseudo-Dice (EMA 0.9)")
    ax1.set_ylim(0, 0.85)
    ax1.set_title("Validation overlap (longest fold)", fontweight="bold")
    ax1.legend(frameon=False, fontsize=9)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("training loss")
    ax2.set_title("Training loss (longest fold)", fontweight="bold")
    fig.suptitle("ADNI-only training fits its training set but never generalises", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "training_sites.png", dpi=160)
    plt.close(fig)


def training_synthseg() -> None:
    ss = _curves()["synthseg"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for tag, col in (("v1", "#805ad5"), ("v2", "#d53f8c")):
        d = ss[tag]
        ep = [e for p, e in zip(d["phase"], d["epoch"]) if p == "dice"]
        lo = [l for p, l in zip(d["phase"], d["loss"]) if p == "dice"]
        ax.plot(ep, lo, "o-", c=col, ms=4, label=f"{tag} (dice phase)")
    ax.axvline(13, c="k", ls="--", lw=1)
    ax.text(13.2, ax.get_ylim()[1] * 0.98, "epoch 13 scored on real T1s:\nDice 0.467 (only checkpoint\nvalidated so far)", va="top", fontsize=9)
    ax.set_xlabel("epoch (1000 synthetic batches each)")
    ax.set_ylabel("soft-Dice training loss")
    ax.set_title("SynthSeg-PC: loss on synthetic images", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "training_synthseg.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    hammers_three_way()
    site_gap()
    training_nnunet()
    training_sites()
    training_synthseg()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
