#!/usr/bin/env python3
"""Score SynthSeg-PC val preds vs GT (PC 1/2 only). Resample-safe.

Writes numbers-only TSV/JSON under tables/. Uses Dice/Jaccard + optional ASD.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "beyondfs" / "modules"))
from mauri_metrics import (  # noqa: E402
    apply_topology_fix,
    average_surface_distance_mm,
    topology_stats,
)


def dice_jaccard(pred: np.ndarray, gold: np.ndarray, lab: int) -> tuple[float, float]:
    p, g = pred == lab, gold == lab
    inter = float(np.logical_and(p, g).sum())
    ps, gs = float(p.sum()), float(g.sum())
    if ps + gs == 0:
        return 1.0, 1.0
    d = 2 * inter / (ps + gs)
    u = ps + gs - inter
    j = inter / u if u > 0 else 1.0
    return d, j


def match_pred(gt_name: str, pred_dir: Path) -> Path | None:
    # GT Hammers_004_lh.nii.gz → pred Hammers_004_lh_0000_synthseg.nii.gz
    stem = gt_name.replace(".nii.gz", "")
    cands = [
        pred_dir / f"{stem}_0000_synthseg.nii.gz",
        pred_dir / f"{stem}_synthseg.nii.gz",
        pred_dir / f"{stem}.nii.gz",
    ]
    for c in cands:
        if c.is_file():
            return c
    # fuzzy
    hits = list(pred_dir.glob(f"{stem}*synthseg*.nii*")) + list(pred_dir.glob(f"{stem}*.nii*"))
    return hits[0] if hits else None


def hemi_labs(gt_name: str) -> list[tuple[int, str]]:
    """Per-hemi FoV crops: only score the side present in the filename."""
    n = gt_name.lower()
    if "_lh" in n:
        return [(1, "L")]
    if "_rh" in n:
        return [(2, "R")]
    return [(1, "L"), (2, "R")]


def score_one(pred_p: Path, gt_p: Path, *, topology_fix: bool = False) -> dict:
    gt_img = nib.load(str(gt_p))
    pr_img = nib.load(str(pred_p))
    if pr_img.shape != gt_img.shape or not np.allclose(pr_img.affine, gt_img.affine, atol=1e-3):
        pr_img = resample_from_to(pr_img, gt_img, order=0)
    gt = np.rint(np.asanyarray(gt_img.dataobj)).astype(np.int16)
    pr = np.rint(np.asanyarray(pr_img.dataobj)).astype(np.int16)
    if topology_fix:
        labs = tuple(lab for lab, _ in hemi_labs(gt_p.name))
        pr = apply_topology_fix(pr, labs if labs else (1, 2))
    zooms = np.asarray(gt_img.header.get_zooms()[:3], dtype=float)
    row: dict = {"case": gt_p.name, "pred": pred_p.name, "topology_fix": topology_fix}
    for lab, side in hemi_labs(gt_p.name):
        # If GT empty for expected hemi → skip (do not count both-empty as Dice=1)
        if not (gt == lab).any():
            row[f"dice_{side}"] = None
            row[f"jaccard_{side}"] = None
            row[f"asd_{side}_mm"] = None
            row[f"vol_pred_{side}"] = int((pr == lab).sum())
            row[f"vol_gt_{side}"] = 0
            row["error"] = f"empty_gt_{side}"
            continue
        d, j = dice_jaccard(pr, gt, lab)
        row[f"dice_{side}"] = round(d, 4)
        row[f"jaccard_{side}"] = round(j, 4)
        row[f"vol_pred_{side}"] = int((pr == lab).sum())
        row[f"vol_gt_{side}"] = int((gt == lab).sum())
        asd = average_surface_distance_mm(pr == lab, gt == lab, zooms)
        row[f"asd_{side}_mm"] = None if asd is None else round(float(asd), 3)
        topo = topology_stats(pr == lab)
        row[f"n_cc_{side}"] = topo["n_cc"]
        row[f"n_holes_{side}"] = topo["n_holes_vox"]
        row[f"largest_cc_frac_{side}"] = topo["largest_cc_frac"]
    return row


def summarize(rows: list[dict]) -> dict:
    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return None if not vals else round(float(np.mean(vals)), 4)

    def n_side(key: str) -> int:
        return sum(1 for r in rows if r.get(key) is not None)

    dL, dR = mean("dice_L"), mean("dice_R")
    # Primary hemi Dice = mean of all scored hemi sides
    sides = [r["dice_L"] for r in rows if r.get("dice_L") is not None] + [
        r["dice_R"] for r in rows if r.get("dice_R") is not None
    ]
    dice_hemi = None if not sides else round(float(np.mean(sides)), 4)

    def band(d: float | None) -> str:
        if d is None:
            return "missing"
        if d >= 0.4:
            return "pass"
        if d <= 0.1:
            return "fail"
        return "inconclusive"

    return {
        "n_cases": len(rows),
        "n_L": n_side("dice_L"),
        "n_R": n_side("dice_R"),
        "dice_L": dL,
        "dice_R": dR,
        "dice_hemi": dice_hemi,
        "jaccard_L": mean("jaccard_L"),
        "jaccard_R": mean("jaccard_R"),
        "asd_L_mm": mean("asd_L_mm"),
        "asd_R_mm": mean("asd_R_mm"),
        "pass_bar_dice": 0.4,
        "fail_bar_dice": 0.1,
        "band_hemi": band(dice_hemi),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", type=Path, required=True, help="folder with *_synthseg.nii.gz")
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data/val_labels")
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument(
        "--topology-fix",
        action="store_true",
        help="Mauri-style largest-CC + fill-holes on pred before Dice/ASD",
    )
    args = ap.parse_args()

    rows = []
    for gt in sorted(args.gt_dir.glob("*.nii.gz")):
        pred = match_pred(gt.name, args.pred_dir)
        if pred is None:
            rows.append({"case": gt.name, "pred": None, "error": "missing_pred"})
            continue
        rows.append(score_one(pred, gt, topology_fix=args.topology_fix))

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # TSV
    keys = [
        "case",
        "dice_L",
        "dice_R",
        "jaccard_L",
        "jaccard_R",
        "asd_L_mm",
        "asd_R_mm",
        "n_cc_L",
        "n_cc_R",
        "n_holes_L",
        "n_holes_R",
        "largest_cc_frac_L",
        "largest_cc_frac_R",
        "vol_pred_L",
        "vol_gt_L",
        "vol_pred_R",
        "vol_gt_R",
        "topology_fix",
        "error",
    ]
    with args.out_prefix.with_suffix(".tsv").open("w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    summary = summarize([r for r in rows if "dice_L" in r])
    summary["pred_dir"] = str(args.pred_dir)
    summary["gt_dir"] = str(args.gt_dir)
    args.out_prefix.with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
