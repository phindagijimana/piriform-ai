#!/usr/bin/env python3
"""Mauri-style extras: mean surface distance, topology, atlas QC.

Numbers only — no image pixels in logs. Used by Hammers / Dataset007 scorers.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage


def _mask(img: nib.spatialimages.SpatialImage, lab: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int32)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    return data == lab, zooms


def average_surface_distance_mm(pred: np.ndarray, gold: np.ndarray, zooms: np.ndarray) -> float | None:
    """Symmetric mean surface distance (mm). None if either mask empty."""
    if not pred.any() or not gold.any():
        return None
    # boundary voxels
    st = ndimage.generate_binary_structure(3, 1)
    pb = pred & ~ndimage.binary_erosion(pred, structure=st, border_value=0)
    gb = gold & ~ndimage.binary_erosion(gold, structure=st, border_value=0)
    if not pb.any():
        pb = pred
    if not gb.any():
        gb = gold
    # distance transform in mm (anisotropic)
    # edt returns distance in voxels; scale approx by mean zoom for speed
    # proper: sample physical coords
    def _pts(mask: np.ndarray) -> np.ndarray:
        ijk = np.argwhere(mask)
        return ijk.astype(float) * zooms.reshape(1, 3)

    pp, gp = _pts(pb), _pts(gb)
    # subsample large surfaces
    rng = np.random.default_rng(0)
    if len(pp) > 2000:
        pp = pp[rng.choice(len(pp), 2000, replace=False)]
    if len(gp) > 2000:
        gp = gp[rng.choice(len(gp), 2000, replace=False)]

    def _mean_nn(a: np.ndarray, b: np.ndarray) -> float:
        # chunked to limit memory
        dmin = np.full(len(a), np.inf)
        for i in range(0, len(b), 500):
            chunk = b[i : i + 500]
            # (Na, Nc)
            d = np.linalg.norm(a[:, None, :] - chunk[None, :, :], axis=2)
            dmin = np.minimum(dmin, d.min(axis=1))
        return float(dmin.mean())

    return round(0.5 * (_mean_nn(pp, gp) + _mean_nn(gp, pp)), 4)


def topology_stats(mask: np.ndarray) -> dict:
    """Connected components + hole voxels (fill_holes − original)."""
    if not mask.any():
        return {"n_cc": 0, "n_holes_vox": 0, "largest_cc_frac": 0.0}
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1)) if n else []
    largest = float(max(sizes) / mask.sum()) if len(sizes) else 0.0
    filled = ndimage.binary_fill_holes(mask)
    holes = int(np.logical_and(filled, ~mask).sum())
    return {"n_cc": int(n), "n_holes_vox": holes, "largest_cc_frac": round(largest, 4)}


def apply_topology_fix(data: np.ndarray, labs: tuple[int, ...] = (1, 2)) -> np.ndarray:
    """Per-label: keep largest CC, fill holes (Mauri fill-holes / remove-islands lite)."""
    out = np.zeros_like(data)
    for lab in labs:
        m = data == lab
        if not m.any():
            continue
        labeled, n = ndimage.label(m)
        if n == 0:
            continue
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        keep = 1 + int(np.argmax(sizes))
        m2 = labeled == keep
        m2 = ndimage.binary_fill_holes(m2)
        out[m2] = lab
    return out


def dice_mask(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return float(2 * inter / s) if s else float("nan")


def atlas_qc_max_dice(
    pred_img: nib.spatialimages.SpatialImage,
    ref_label_paths: list[Path],
    pred_lab: int,
    ref_lab: int,
    max_refs: int = 30,
) -> dict:
    """Max Dice of pred vs reference manuals after centroid translation (no full ANTs).

    Mauri uses MNI + SynthMorph; this is a cheap same-grid / translate proxy for
    Hammers-space QC until a true MNI prior exists.
    """
    pred = np.rint(np.asanyarray(pred_img.dataobj)).astype(np.int32) == pred_lab
    if not pred.any():
        return {"atlas_qc_max_dice": None, "atlas_qc_best_ref": None, "atlas_qc_n_refs": 0}
    p_idx = np.argwhere(pred)
    p_cent = p_idx.mean(0)
    best, best_name = -1.0, None
    n = 0
    for ref_p in ref_label_paths[:max_refs]:
        if not ref_p.is_file():
            continue
        r_img = nib.load(str(ref_p))
        r_img = resample_from_to(r_img, pred_img, order=0)
        ref = np.rint(np.asanyarray(r_img.dataobj)).astype(np.int32) == ref_lab
        if not ref.any():
            continue
        r_cent = np.argwhere(ref).mean(0)
        shift = np.round(p_cent - r_cent).astype(int)
        shifted = np.zeros_like(ref)
        # apply integer shift
        src = ref
        dst = shifted
        slices_src = []
        slices_dst = []
        ok = True
        for ax, s in enumerate(shift):
            if s >= 0:
                slices_src.append(slice(0, src.shape[ax] - s))
                slices_dst.append(slice(s, src.shape[ax]))
            else:
                slices_src.append(slice(-s, src.shape[ax]))
                slices_dst.append(slice(0, src.shape[ax] + s))
            if slices_src[-1].start >= slices_src[-1].stop:
                ok = False
                break
        if not ok:
            continue
        dst[tuple(slices_dst)] = src[tuple(slices_src)]
        d = dice_mask(pred, dst)
        n += 1
        if not np.isnan(d) and d > best:
            best, best_name = d, ref_p.name
    return {
        "atlas_qc_max_dice": None if best < 0 else round(best, 4),
        "atlas_qc_best_ref": best_name,
        "atlas_qc_n_refs": n,
    }


def side_extras(
    pred_img: nib.spatialimages.SpatialImage,
    gold_img: nib.spatialimages.SpatialImage | None,
    pred_lab: int,
    gold_lab: int | None = None,
) -> dict:
    pm, zooms = _mask(pred_img, pred_lab)
    topo = topology_stats(pm)
    out = {
        "asd_mm": None,
        "n_cc": topo["n_cc"],
        "n_holes_vox": topo["n_holes_vox"],
        "largest_cc_frac": topo["largest_cc_frac"],
    }
    if gold_img is not None and gold_lab is not None:
        gm, _ = _mask(gold_img, gold_lab)
        out["asd_mm"] = average_surface_distance_mm(pm, gm, zooms)
    return out
