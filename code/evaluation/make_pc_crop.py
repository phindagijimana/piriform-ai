#!/usr/bin/env python3
"""60 mm T1 crops for Mauri-style inference prior-crop (Dataset002/007).

Prior modes: jlf / atlas_pc (125/126 centroids), putamen (aseg 12/51).
Used by predict_nnunet_prior_crop_job.sh. Does not retrain. Numbers only in logs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

PC = {"lh": 125, "rh": 126}
PC_NNUNET = {"lh": 1, "rh": 2}  # ADNI Dataset005 / remapped manuals
PUTAMEN = {"lh": 12, "rh": 51}  # FreeSurfer aseg
CROP_HELPER_DEFAULT = Path(
    "/path/to/clau_seg/"
    "tools/bin/crop_to_fov"
)


def _pc_labmap(prior: nib.spatialimages.SpatialImage, prior_mode: str) -> dict:
    """Pick 125/126 vs 1/2 from the prior volume (oracle ADNI = 1/2)."""
    if prior_mode == "oracle":
        data = np.rint(np.asanyarray(prior.dataobj)).astype(np.int16)
        if int((data == 1).sum()) + int((data == 2).sum()) > 0:
            return PC_NNUNET
        if int((data == 125).sum()) + int((data == 126).sum()) > 0:
            return PC
        raise SystemExit("oracle prior has neither 1/2 nor 125/126")
    return PC


def sphere_mask(shape, affine, center_ras, radius_mm=8.0) -> np.ndarray:
    ijk = np.array(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"))
    ijk = np.stack(ijk, axis=-1).reshape(-1, 3)
    xyz = nib.affines.apply_affine(affine, ijk)
    keep = np.linalg.norm(xyz - center_ras, axis=1) <= radius_mm
    out = np.zeros(shape, dtype=np.uint8)
    out.reshape(-1)[keep] = 1
    return out


def centroid_ras(img: nib.spatialimages.SpatialImage, lab: int) -> np.ndarray:
    data = np.asanyarray(img.dataobj)
    idx = np.argwhere(data == lab)
    if idx.size == 0:
        raise SystemExit(f"missing label {lab}")
    return nib.affines.apply_affine(img.affine, idx.mean(0))


def write_crops(
    t1_p: Path,
    prior_p: Path,
    out_dir: Path,
    crop_helper: Path,
    fov_mm: float = 60.0,
    prior_mode: str = "jlf",
    putamen_shift_mm: tuple[float, float] = (0.0, 0.0),
) -> dict:
    """prior_mode: jlf/atlas_pc (125/126), oracle (gold 1/2 or 125/126), putamen."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t1 = nib.load(str(t1_p))
    prior = nib.load(str(prior_p))
    if prior_mode == "putamen":
        labmap = PUTAMEN
        cite = "Mauri put0-style: aseg putamen centroid → 60 mm FoV (not gold)"
    elif prior_mode == "oracle":
        labmap = _pc_labmap(prior, prior_mode)
        cite = "oracle FoV: gold PC centroid (ablation; not a deployable prior)"
    else:
        labmap = PC
        cite = "Lyu 2023 / Mauri 2025 crop; prior=JLF or atlas PC (not gold for accuracy)"
    lat, zmm = putamen_shift_mm
    meta: dict = {
        "t1": str(t1_p),
        "prior": str(prior_p),
        "prior_mode": prior_mode,
        "fov_mm": fov_mm,
        "cite": cite,
        "hemis": {},
    }
    for hemi, lab in labmap.items():
        ras = centroid_ras(prior, lab)
        if prior_mode == "putamen":
            # match make_putamen_fov: lh x-=lat, rh x+=lat, z+=zmm
            ras = ras + np.array([-lat if hemi == "lh" else lat, 0.0, zmm])
        mask = sphere_mask(t1.shape[:3], t1.affine, ras)
        mask_p = out_dir / f"prior_blob.{hemi}.nii.gz"
        nib.save(nib.Nifti1Image(mask, t1.affine, t1.header), str(mask_p))
        crop_p = out_dir / f"{hemi}.crop.nii.gz"
        r = subprocess.run(
            [
                sys.executable,
                str(crop_helper),
                str(t1_p),
                str(mask_p),
                str(crop_p),
                "--threshold",
                "0.5",
                "--fov-mm",
                str(fov_mm),
                str(fov_mm),
                str(fov_mm),
            ],
            check=False,
        )
        if r.returncode != 0:
            raise SystemExit(f"crop_to_fov failed for {hemi} rc={r.returncode}")
        meta["hemis"][hemi] = {
            "prior_label": lab,
            "centroid_ras": [float(x) for x in np.round(ras, 3)],
            "crop": str(crop_p),
        }
        print(
            f"{hemi} {prior_mode}{lab} RAS={np.round(ras, 2).tolist()} crop={crop_p}",
            flush=True,
        )
    (out_dir / "crop_centres.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def stitch(
    t1_p: Path,
    lh_pred: Path,
    rh_pred: Path,
    out_p: Path,
) -> dict:
    target = nib.load(str(t1_p))
    canvas = np.zeros(target.shape[:3], dtype=np.int16)
    counts = {}
    for pred_p, keep, other in (
        (lh_pred, 1, 2),
        (rh_pred, 2, 1),
    ):
        pred = nib.load(str(pred_p))
        on_t = resample_from_to(pred, target, order=0)
        d = np.rint(np.asanyarray(on_t.dataobj)).astype(np.int16)
        n_keep = int((d == keep).sum())
        n_other = int((d == other).sum())
        counts[pred_p.name] = {f"class_{keep}": n_keep, f"class_{other}": n_other}
        canvas[d == keep] = keep
        leftover = (d == other) & (canvas == 0)
        canvas[leftover] = other
    zo = float(np.prod(target.header.get_zooms()[:3]))
    n1 = int((canvas == 1).sum())
    n2 = int((canvas == 2).sum())
    out_p.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(canvas, target.affine, target.header)
    img.set_data_dtype(np.int16)
    nib.save(img, str(out_p))
    summary = {
        "out": str(out_p),
        "n_L": n1,
        "n_R": n2,
        "mm3_L": round(n1 * zo, 1),
        "mm3_R": round(n2 * zo, 1),
        "crop_counts": counts,
    }
    print(
        f"stitched {out_p} L={n1} ({summary['mm3_L']} mm3) "
        f"R={n2} ({summary['mm3_R']} mm3)",
        flush=True,
    )
    (out_p.with_suffix(".json")).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def self_test() -> int:
    aff = np.eye(4)
    aff[0, 0] = aff[1, 1] = aff[2, 2] = 1.0
    t1 = np.ones((40, 40, 40), dtype=np.float32)
    jlf = np.zeros((40, 40, 40), dtype=np.int16)
    jlf[10, 20, 20] = 125
    jlf[30, 20, 20] = 126
    helper = CROP_HELPER_DEFAULT
    if not helper.is_file():
        print("SKIP self-test: crop_to_fov missing", file=sys.stderr)
        return 0
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        t1_p = root / "t1.nii.gz"
        jlf_p = root / "jlf.nii.gz"
        nib.save(nib.Nifti1Image(t1, aff), str(t1_p))
        nib.save(nib.Nifti1Image(jlf, aff), str(jlf_p))
        meta = write_crops(t1_p, jlf_p, root / "fov", helper, fov_mm=12.0, prior_mode="jlf")
        lh = np.array(meta["hemis"]["lh"]["centroid_ras"])
        rh = np.array(meta["hemis"]["rh"]["centroid_ras"])
        assert abs(lh[0] - 10.0) < 0.6, meta
        assert abs(rh[0] - 30.0) < 0.6, meta
        # fake hemi preds: 1 on left crop, 2 on right crop
        lh_img = nib.load(str(root / "fov/lh.crop.nii.gz"))
        rh_img = nib.load(str(root / "fov/rh.crop.nii.gz"))
        lp = np.zeros(lh_img.shape[:3], dtype=np.int16)
        rp = np.zeros(rh_img.shape[:3], dtype=np.int16)
        lp[tuple(s // 2 for s in lp.shape)] = 1
        rp[tuple(s // 2 for s in rp.shape)] = 2
        lp_p = root / "lh_pred.nii.gz"
        rp_p = root / "rh_pred.nii.gz"
        nib.save(nib.Nifti1Image(lp, lh_img.affine, lh_img.header), str(lp_p))
        nib.save(nib.Nifti1Image(rp, rh_img.affine, rh_img.header), str(rp_p))
        st = stitch(t1_p, lp_p, rp_p, root / "st.nii.gz")
        assert st["n_L"] >= 1 and st["n_R"] >= 1, st
    print("self-test ok")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--t1", type=Path)
    p.add_argument("--jlf", type=Path, help="JLF or atlas PC labels (125/126)")
    p.add_argument("--aseg", type=Path, help="FreeSurfer aseg for putamen prior")
    p.add_argument(
        "--prior-mode",
        choices=("jlf", "putamen", "atlas_pc", "oracle"),
        default="jlf",
    )
    p.add_argument("--putamen-lateral-mm", type=float, default=0.0)
    p.add_argument("--putamen-z-mm", type=float, default=0.0)
    p.add_argument("--out", type=Path)
    p.add_argument("--fov-mm", type=float, default=60.0)
    p.add_argument("--crop-helper", type=Path, default=CROP_HELPER_DEFAULT)
    p.add_argument("--stitch", action="store_true")
    p.add_argument("--lh-pred", type=Path)
    p.add_argument("--rh-pred", type=Path)
    p.add_argument("--stitched", type=Path)
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if args.stitch:
        if not args.t1 or not args.lh_pred or not args.rh_pred or not args.stitched:
            p.error("--stitch needs --t1 --lh-pred --rh-pred --stitched")
        stitch(args.t1, args.lh_pred, args.rh_pred, args.stitched)
        return 0
    if not args.t1 or not args.out:
        p.error("--t1 --out required (or --self-test / --stitch)")
    mode = args.prior_mode
    if mode == "putamen":
        prior = args.aseg
        if not prior:
            p.error("--prior-mode putamen needs --aseg")
    else:
        prior = args.jlf
        if not prior:
            p.error("--prior-mode jlf/atlas_pc/oracle needs --jlf (gold or warped prior)")
    if not args.crop_helper.is_file():
        print(f"ERROR: crop helper missing {args.crop_helper}", file=sys.stderr)
        return 1
    write_crops(
        args.t1,
        prior,
        args.out,
        args.crop_helper,
        args.fov_mm,
        prior_mode=mode,
        putamen_shift_mm=(args.putamen_lateral_mm, args.putamen_z_mm),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
