#!/usr/bin/env python3
"""Hammers Stage A table: nnU-Net 5-fold vs manuals, plus MAPER/JLF if present.

Jaccard and |ΔV| mm³, left/right separate (Steinbart 2023). Does not print voxels.
Does not mix ADNI. Does not treat literature 0.50 as a measured column.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

ROOT = Path("/path/to/piriform")
RAW = ROOT / "nnunet/nnUNet_raw/Dataset001_Piriform"
PREPROC = ROOT / "nnunet/nnUNet_preprocessed/Dataset001_Piriform"
RESULTS = (
    ROOT
    / "nnunet/nnUNet_results/Dataset001_Piriform"
    / "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
)
ATLAS = ROOT / "Piriform_Segmentation/atlas"
MALF = ROOT / "Piriform_Segmentation/malf"
OUT_DEFAULT = ROOT / "nnunet/tables/hammers_stage_a.tsv"

# nnU-Net Dataset001: 1=left, 2=right. MALF/MAPER: 125/126.
LAB = {"nnunet": (1, 2), "jlf": (125, 126), "maper": (125, 126), "gold": (1, 2)}


def load_splits() -> list[dict]:
    p = PREPROC / "splits_final.json"
    return json.loads(p.read_text())


def hammers_to_a02(case: str) -> str:
    n = int(case.split("_")[1])
    return f"a{n:02d}"


def hammers_to_maper_id(case: str) -> str:
    n = int(case.split("_")[1])
    return f"a{n}"


def pred_paths(case: str) -> dict[str, Path | None]:
    aid = hammers_to_a02(case)
    cross = RESULTS / "crossval_results_folds_0_1_2_3_4"
    post = cross / "postprocessed" / f"{case}.nii.gz"
    cross_raw = cross / f"{case}.nii.gz"
    fold_hit: Path | None = None
    for fold in range(5):
        p = RESULTS / f"fold_{fold}" / "validation" / f"{case}.nii.gz"
        if p.is_file():
            fold_hit = p
            break
    nnunet = post if post.is_file() else cross_raw if cross_raw.is_file() else fold_hit
    jlf = MALF / aid / f"{aid}_Labels.nii.gz"
    maper = MALF / f"{aid}-MAPER-r95pc" / f"{aid}_MAPER_Labels.nii.gz"
    gold = RAW / "labelsTr" / f"{case}.nii.gz"
    return {
        "nnunet": nnunet if nnunet and nnunet.is_file() else None,
        "jlf": jlf if jlf.is_file() else None,
        "maper": maper if maper.is_file() else None,
        "gold": gold if gold.is_file() else None,
    }


def as_mask(img: nib.spatialimages.SpatialImage, lab: int) -> tuple[np.ndarray, float]:
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int32)
    zo = float(np.prod(img.header.get_zooms()[:3]))
    return data == lab, zo


def metrics(
    pred: Path,
    gold: Path,
    pred_labs: tuple[int, int],
    *,
    mauri_extras: bool = True,
    topology_fix: bool = False,
) -> dict:
    from mauri_metrics import apply_topology_fix, side_extras  # local module

    g_img = nib.load(str(gold))
    p_img = nib.load(str(pred))
    if p_img.shape[:3] != g_img.shape[:3] or not np.allclose(p_img.affine, g_img.affine, atol=1e-3):
        p_img = resample_from_to(p_img, g_img, order=0)
    if topology_fix:
        pdata = apply_topology_fix(np.rint(np.asanyarray(p_img.dataobj)).astype(np.int16), pred_labs)
        p_img = nib.Nifti1Image(pdata, p_img.affine, p_img.header)
    out = {}
    for side, gl, pl in (("L", 1, pred_labs[0]), ("R", 2, pred_labs[1])):
        gm, gzo = as_mask(g_img, gl)
        pm, pzo = as_mask(p_img, pl)
        inter = int(np.logical_and(gm, pm).sum())
        union = int(np.logical_or(gm, pm).sum())
        gs = int(gm.sum())
        ps = int(pm.sum())
        jacc = (inter / union) if union else float("nan")
        dice = (2 * inter / (gs + ps)) if (gs + ps) else float("nan")
        vg = gs * gzo
        vp = ps * pzo
        out[f"jaccard_{side}"] = None if np.isnan(jacc) else round(jacc, 4)
        out[f"dice_{side}"] = None if np.isnan(dice) else round(dice, 4)
        out[f"vol_gt_{side}_mm3"] = round(vg, 1)
        out[f"vol_pred_{side}_mm3"] = round(vp, 1)
        out[f"dvol_{side}_mm3"] = round(abs(vp - vg), 1)
        if mauri_extras:
            ex = side_extras(p_img, g_img, pl, gl)
            out[f"asd_{side}_mm"] = ex["asd_mm"]
            out[f"n_cc_{side}"] = ex["n_cc"]
            out[f"n_holes_{side}"] = ex["n_holes_vox"]
            out[f"largest_cc_frac_{side}"] = ex["largest_cc_frac"]
    return out


def mean_col(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return round(float(np.mean(vals)), 4)


def run_table(out_tsv: Path) -> dict:
    splits = load_splits()
    case_fold = {}
    for i, sp in enumerate(splits):
        for c in sp["val"]:
            case_fold[c] = i
    rows = []
    missing = {"nnunet": [], "jlf": [], "maper": []}
    for case in sorted(case_fold):
        paths = pred_paths(case)
        gold = paths["gold"]
        if gold is None:
            raise SystemExit(f"missing gold {case}")
        for method in ("nnunet", "jlf", "maper"):
            src = paths[method]
            if src is None:
                missing[method].append(case)
                continue
            row = {
                "case": case,
                "fold": case_fold[case],
                "method": method,
                "atlas": hammers_to_a02(case),
                "pred": str(src),
            }
            row.update(metrics(src, gold, LAB[method]))
            rows.append(row)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "fold",
        "method",
        "atlas",
        "jaccard_L",
        "jaccard_R",
        "dice_L",
        "dice_R",
        "asd_L_mm",
        "asd_R_mm",
        "vol_gt_L_mm3",
        "vol_pred_L_mm3",
        "dvol_L_mm3",
        "vol_gt_R_mm3",
        "vol_pred_R_mm3",
        "dvol_R_mm3",
        "pred",
    ]
    with out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary: dict = {"n_rows": len(rows), "tsv": str(out_tsv), "missing": missing}
    print(f"wrote {out_tsv} rows={len(rows)}")
    print("Steinbart 2023 MAPER bar (literature, not this run): Jaccard ~0.50 controls")
    for method in ("nnunet", "jlf", "maper"):
        sub = [r for r in rows if r["method"] == method]
        jl = mean_col(sub, "jaccard_L")
        jr = mean_col(sub, "jaccard_R")
        dl = mean_col(sub, "dice_L")
        dr = mean_col(sub, "dice_R")
        asdl = mean_col(sub, "asd_L_mm")
        asdr = mean_col(sub, "asd_R_mm")
        dvl = mean_col(sub, "dvol_L_mm3")
        dvr = mean_col(sub, "dvol_R_mm3")
        summary[method] = {
            "n": len(sub),
            "jaccard_L": jl,
            "jaccard_R": jr,
            "dice_L": dl,
            "dice_R": dr,
            "asd_L_mm": asdl,
            "asd_R_mm": asdr,
            "dvol_L_mm3": dvl,
            "dvol_R_mm3": dvr,
        }
        miss = len(missing[method])
        print(
            f"{method}: n={len(sub)} missing={miss} "
            f"Jaccard L/R={jl}/{jr} Dice L/R={dl}/{dr} "
            f"ASD L/R={asdl}/{asdr} mm |ΔV| L/R={dvl}/{dvr} mm3"
        )
    (out_tsv.with_suffix(".json")).write_text(json.dumps(summary, indent=2) + "\n")
    if missing["nnunet"]:
        raise SystemExit(f"ERROR: nnU-Net val preds missing: {missing['nnunet']}")
    return summary


def self_test() -> int:
    aff = np.eye(4, dtype=float)
    aff[0, 0] = aff[1, 1] = aff[2, 2] = 1.0
    gold = np.zeros((8, 8, 8), dtype=np.int16)
    pred = np.zeros((8, 8, 8), dtype=np.int16)
    gold[2:5, 2:5, 2:5] = 1
    pred[2:5, 2:5, 2:4] = 1  # miss one z-slice of the 3x3x3 cube
    gold[5:7, 5:7, 5:7] = 2
    pred[5:7, 5:7, 5:7] = 2
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        g = root / "g.nii.gz"
        p = root / "p.nii.gz"
        nib.save(nib.Nifti1Image(gold, aff), str(g))
        nib.save(nib.Nifti1Image(pred, aff), str(p))
        m = metrics(p, g, (1, 2))
        # left: inter=18, union=27, jaccard=18/27=0.666..., dice=36/45=0.8
        assert abs(m["jaccard_L"] - 0.6667) < 1e-3, m
        assert abs(m["dice_L"] - 0.8) < 1e-3, m
        assert m["jaccard_R"] == 1.0, m
        assert m["dvol_L_mm3"] == 9.0, m
        assert m["dvol_R_mm3"] == 0.0, m
    print("self-test ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    run_table(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
