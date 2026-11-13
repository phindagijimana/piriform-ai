#!/usr/bin/env python3
"""Fail-closed FoV QC for piriform prior-crop (claustrum-style gate).

Checks (numbers only — no pixels in logs):
  - prior→PC distance (mm) per hemi
  - fraction of JLF PC (125/126) inside each hemi crop
  - crop intensity (nonzero mean)
  - pred volume (labels 1/2)
  - pass/fail with reasons

Piriform is ~45 mm from putamen; a 60 mm putamen crop does **not** contain PC
when T1 and aseg share space. Prefer prior_mode=jlf|atlas_pc for PC infer.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# Claustrum used 25 mm putamen gate; PC sits ~45 mm from putamen → putamen FoV fails.
MAX_PRIOR_TO_PC_MM = 25.0
MIN_PC_IN_CROP_FRAC = 0.50
MIN_PRED_VOX = 50
MIN_CROP_MEAN = 10.0  # soft; empty air crops fail harder via PC frac


def _centroid(img: nib.spatialimages.SpatialImage, lab: int) -> np.ndarray | None:
    d = np.asanyarray(img.dataobj)
    idx = np.argwhere(d == lab)
    if idx.size == 0:
        return None
    return nib.affines.apply_affine(img.affine, idx.mean(0))


def _load_on(target: nib.spatialimages.SpatialImage, path: Path) -> np.ndarray:
    img = nib.load(str(path))
    if img.shape[:3] == target.shape[:3] and np.allclose(img.affine, target.affine, atol=1e-2):
        return np.rint(np.asanyarray(img.dataobj)).astype(np.int16)
    return np.rint(np.asanyarray(resample_from_to(img, target, order=0).dataobj)).astype(np.int16)


def _pc_frac_in_crop(
    t1: nib.spatialimages.SpatialImage,
    gold: np.ndarray,
    crop_p: Path,
    lab: int,
) -> float:
    cimg = nib.load(str(crop_p))
    ones = nib.Nifti1Image(np.ones(cimg.shape[:3], np.uint8), cimg.affine, cimg.header)
    m = np.asanyarray(resample_from_to(ones, t1, order=0).dataobj) > 0
    n = int((gold == lab).sum())
    if n == 0:
        return 0.0
    return float(((gold == lab) & m).sum() / n)


def qc_one(
    *,
    subject: str,
    crop_root: Path,
    jlf: Path,
    pred: Path | None,
    max_prior_pc_mm: float,
    min_pc_frac: float,
    min_pred_vox: int,
) -> dict:
    meta = json.loads((crop_root / "fov" / "crop_centres.json").read_text())
    t1 = nib.load(meta["t1"])
    gold = _load_on(t1, jlf)
    prior_mode = meta.get("prior_mode", "")
    flags: list[str] = []

    row: dict = {
        "subject": subject,
        "prior_mode": prior_mode,
        "t1": meta["t1"],
        "jlf": str(jlf),
        "crop_root": str(crop_root),
    }

    # Gold PC IDs: JLF 125/126, ADNI manuals 1/2
    if int((gold == 125).sum()) + int((gold == 126).sum()) > 0:
        pc_labs = {"lh": 125, "rh": 126}
    elif int((gold == 1).sum()) + int((gold == 2).sum()) > 0:
        pc_labs = {"lh": 1, "rh": 2}
    else:
        pc_labs = {"lh": 125, "rh": 126}
        flags.append("no_gold_pc_labels")

    # prior→PC distance from crop centres vs gold PC centroids
    for hemi, pc_lab in pc_labs.items():
        ras = np.array(meta["hemis"][hemi]["centroid_ras"], dtype=float)
        pc = _centroid(nib.Nifti1Image(gold, t1.affine, t1.header), pc_lab)
        dist = float(np.linalg.norm(ras - pc)) if pc is not None else float("nan")
        row[f"prior_to_pc_mm_{hemi}"] = None if np.isnan(dist) else round(dist, 2)
        crop_p = crop_root / "fov" / f"{hemi}.crop.nii.gz"
        frac = _pc_frac_in_crop(t1, gold, crop_p, pc_lab) if crop_p.is_file() else 0.0
        row[f"pc_frac_in_crop_{hemi}"] = round(frac, 4)
        v = np.asanyarray(nib.load(str(crop_p)).dataobj).astype(float) if crop_p.is_file() else np.array([0.0])
        row[f"crop_mean_{hemi}"] = round(float(np.mean(v)), 2)
        if dist > max_prior_pc_mm:
            flags.append(f"prior_pc_{hemi}>{max_prior_pc_mm:.0f}mm")
        if frac < min_pc_frac:
            flags.append(f"pc_outside_crop_{hemi}")

    nL = nR = 0
    if pred is not None and pred.is_file():
        pdata = _load_on(t1, pred)
        # pred may be 1/2 already on native
        if not np.allclose(nib.load(str(pred)).affine, t1.affine, atol=1e-2):
            pdata = _load_on(t1, pred)
        else:
            pdata = np.rint(np.asanyarray(nib.load(str(pred)).dataobj)).astype(np.int16)
        nL, nR = int((pdata == 1).sum()), int((pdata == 2).sum())
    row["pred_vox_L"] = nL
    row["pred_vox_R"] = nR
    if nL < min_pred_vox:
        flags.append("empty_pred_L")
    if nR < min_pred_vox:
        flags.append("empty_pred_R")

    # Hard fail if either hemi has PC mostly outside crop
    fov_fail = any(row[f"pc_frac_in_crop_{h}"] < min_pc_frac for h in ("lh", "rh"))
    empty_fail = nL < min_pred_vox or nR < min_pred_vox
    row["fov_pass"] = not fov_fail
    row["pred_pass"] = not empty_fail
    row["pass"] = (not fov_fail) and (not empty_fail)
    row["flags"] = ",".join(flags) if flags else ""
    if prior_mode == "putamen" and fov_fail:
        row["note"] = "putamen FoV is for claustrum; use jlf/atlas_pc for piriform"
    else:
        row["note"] = ""
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop-root", type=Path, required=True)
    ap.add_argument("--jlf", type=Path, required=True)
    ap.add_argument("--pred", type=Path, default=None)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-prior-pc-mm", type=float, default=MAX_PRIOR_TO_PC_MM)
    ap.add_argument("--min-pc-frac", type=float, default=MIN_PC_IN_CROP_FRAC)
    ap.add_argument("--min-pred-vox", type=int, default=MIN_PRED_VOX)
    args = ap.parse_args()

    pred = args.pred
    if pred is None:
        for cand in (
            args.crop_root / "pred_native" / f"{args.subject}_priorcrop_topo.nii.gz",
            args.crop_root / "pred_native" / f"{args.subject}_priorcrop.nii.gz",
        ):
            if cand.is_file():
                pred = cand
                break

    row = qc_one(
        subject=args.subject,
        crop_root=args.crop_root,
        jlf=args.jlf,
        pred=pred,
        max_prior_pc_mm=args.max_prior_pc_mm,
        min_pc_frac=args.min_pc_frac,
        min_pred_vox=args.min_pred_vox,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    exists = args.out.is_file()
    with args.out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(json.dumps({k: row[k] for k in ("subject", "prior_mode", "pass", "fov_pass", "pred_pass", "flags", "note")}, indent=2))
    return 0 if row["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
