#!/usr/bin/env python3
"""Score prior-crop pred (labels 1/2) vs official JLF (125/126).

Reports Dice, Jaccard, |ΔV| mm³, and ASD (mm) — Steinbart/Mauri-style, not Dice-only.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# beyondfs/modules on path when invoked from jobs
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mauri_metrics import side_extras  # noqa: E402


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    inter = int(np.logical_and(a, b).sum())
    s = int(a.sum()) + int(b.sum())
    if s == 0:
        return None
    return round(2 * inter / s, 4)


def jaccard(a: np.ndarray, b: np.ndarray) -> float | None:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return round(inter / union, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--jlf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--prior-mode", default="")
    ap.add_argument("--folds", default="")
    args = ap.parse_args()

    g_img = nib.load(str(args.jlf))
    p_img = nib.load(str(args.pred))
    if p_img.shape[:3] != g_img.shape[:3] or not np.allclose(p_img.affine, g_img.affine, atol=1e-3):
        p_img = resample_from_to(p_img, g_img, order=0)
    pred = np.rint(np.asanyarray(p_img.dataobj)).astype(np.int16)
    gold = np.rint(np.asanyarray(g_img.dataobj)).astype(np.int16)
    zo = float(np.prod(g_img.header.get_zooms()[:3]))
    # JLF = 125/126; ADNI Dataset005 manuals = 1/2
    if int((gold == 125).sum()) + int((gold == 126).sum()) > 0:
        gold_lr = (125, 126)
    elif int((gold == 1).sum()) + int((gold == 2).sum()) > 0:
        gold_lr = (1, 2)
    else:
        raise SystemExit(f"no PC labels 125/126 or 1/2 in gold {args.jlf}")

    row = {
        "subject": args.subject,
        "dataset": args.dataset,
        "prior_mode": args.prior_mode,
        "folds": args.folds,
        "pred": str(args.pred),
        "jlf": str(args.jlf),
        "gold_labels": f"{gold_lr[0]}/{gold_lr[1]}",
    }
    for side, pl, gl in (("L", 1, gold_lr[0]), ("R", 2, gold_lr[1])):
        pm, gm = pred == pl, gold == gl
        vp, vg = int(pm.sum()), int(gm.sum())
        row[f"dice_{side}"] = dice(pm, gm)
        row[f"jaccard_{side}"] = jaccard(pm, gm)
        row[f"vol_pred_{side}"] = vp
        row[f"vol_jlf_{side}"] = vg
        row[f"vol_pred_{side}_mm3"] = round(vp * zo, 1)
        row[f"vol_jlf_{side}_mm3"] = round(vg * zo, 1)
        row[f"dvol_{side}_mm3"] = round(abs(vp - vg) * zo, 1)
        ex = side_extras(p_img, g_img, pl, gl)
        row[f"asd_{side}_mm"] = None if ex["asd_mm"] is None else round(float(ex["asd_mm"]), 4)

    dice_sides = [v for v in (row["dice_L"], row["dice_R"]) if v is not None]
    jacc_sides = [v for v in (row["jaccard_L"], row["jaccard_R"]) if v is not None]
    asd_sides = [v for v in (row["asd_L_mm"], row["asd_R_mm"]) if v is not None]
    row["dice_hemi"] = None if not dice_sides else round(float(np.mean(dice_sides)), 4)
    row["jaccard_hemi"] = None if not jacc_sides else round(float(np.mean(jacc_sides)), 4)
    row["asd_hemi_mm"] = None if not asd_sides else round(float(np.mean(asd_sides)), 4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    exists = args.out.is_file()
    with args.out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
