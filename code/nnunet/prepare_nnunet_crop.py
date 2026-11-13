#!/usr/bin/env python3
"""Hammers 60 mm PC crops for Dataset002 (gold manuals, not JLF output).

Per-hemisphere 60 mm FoV around Steinbart 125/126. Same locked 5-fold
membership as Dataset001 (lh/rh of one brain stay in the same split).
Does not overwrite Dataset001. Does not print voxels.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path("/path/to/piriform")
ATLAS = ROOT / "Piriform_Segmentation/atlas"
RAW1 = ROOT / "nnunet/nnUNet_raw/Dataset001_Piriform"
OUT = ROOT / "nnunet/nnUNet_raw/Dataset002_PiriformCrop60"
LOCKED1 = ROOT / "nnunet/splits/Dataset001_Piriform.json"
LOCKED2 = ROOT / "nnunet/splits/Dataset002_PiriformCrop60.json"
CROP_HELPER = Path(
    "/path/to/clau_seg/tools/bin/crop_to_fov"
)
PC = {"lh": 125, "rh": 126}
FOV_MM = 60.0


def sphere_mask(shape, affine, center_ras, radius_mm=8.0) -> np.ndarray:
    ijk = np.array(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"))
    ijk = np.stack(ijk, axis=-1).reshape(-1, 3)
    xyz = nib.affines.apply_affine(affine, ijk)
    keep = np.linalg.norm(xyz - center_ras, axis=1) <= radius_mm
    out = np.zeros(shape, dtype=np.uint8)
    out.reshape(-1)[keep] = 1
    return out


def remap_crop(src: Path, dst: Path) -> tuple[int, int]:
    img = nib.load(str(src))
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int16)
    out = np.zeros(data.shape, dtype=np.int16)
    out[data == 125] = 1
    out[data == 126] = 2
    extra = np.setdiff1d(np.unique(data), np.array([0, 125, 126]))
    if extra.size:
        raise SystemExit(f"{src.name}: unexpected labels {extra.tolist()}")
    hdr = img.header.copy()
    hdr.set_data_dtype(np.int16)
    nib.Nifti1Image(out, img.affine, hdr).to_filename(str(dst))
    return int((out == 1).sum()), int((out == 2).sum())


def crop_pair(brain: Path, labels: Path, ras: np.ndarray, dest_brain: Path, dest_lab: Path) -> None:
    tmp_mask = dest_brain.with_name(dest_brain.name + ".prior.nii.gz")
    img = nib.load(str(brain))
    mask = sphere_mask(img.shape[:3], img.affine, ras)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(tmp_mask))
    for src, dst in ((brain, dest_brain), (labels, dest_lab)):
        r = subprocess.run(
            [
                sys.executable,
                str(CROP_HELPER),
                str(src),
                str(tmp_mask),
                str(dst),
                "--threshold",
                "0.5",
                "--fov-mm",
                str(FOV_MM),
                str(FOV_MM),
                str(FOV_MM),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise SystemExit(f"crop_to_fov failed {src} -> {dst}: {r.stderr}")
    tmp_mask.unlink(missing_ok=True)


def expand_splits() -> None:
    splits = json.loads(LOCKED1.read_text())
    out = []
    for fold in splits:
        out.append(
            {
                "train": sorted(f"{c}_{h}" for c in fold["train"] for h in ("lh", "rh")),
                "val": sorted(f"{c}_{h}" for c in fold["val"] for h in ("lh", "rh")),
            }
        )
    LOCKED2.parent.mkdir(parents=True, exist_ok=True)
    LOCKED2.write_text(json.dumps(out, indent=4) + "\n")
    print(f"wrote locked splits {LOCKED2}")
    for i, fold in enumerate(out):
        print(f"  fold {i}: train={len(fold['train'])} val={len(fold['val'])}")


def main() -> int:
    if not CROP_HELPER.is_file():
        raise SystemExit(f"missing {CROP_HELPER}")
    images = OUT / "imagesTr"
    labels_dir = OUT / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for i in range(1, 31):
        aid = f"a{i:02d}"
        case0 = f"Hammers_{i:03d}"
        brain = ATLAS / f"{aid}_BrainExtractionBrain.nii.gz"
        lab = ATLAS / f"{aid}_PC_labels.nii.gz"
        gold = nib.load(str(lab))
        gd = np.asanyarray(gold.dataobj)
        for hemi, labid in PC.items():
            idx = np.argwhere(gd == labid)
            if idx.size == 0:
                raise SystemExit(f"{aid} missing {labid}")
            ras = nib.affines.apply_affine(gold.affine, idx.mean(0))
            case = f"{case0}_{hemi}"
            tmp_lab = labels_dir / f"{case}_125126.nii.gz"
            dest_b = images / f"{case}_0000.nii.gz"
            dest_l = labels_dir / f"{case}.nii.gz"
            crop_pair(brain, lab, ras, dest_b, tmp_lab)
            n1, n2 = remap_crop(tmp_lab, dest_l)
            tmp_lab.unlink(missing_ok=True)
            need = 1 if hemi == "lh" else 2
            got = n1 if hemi == "lh" else n2
            if got == 0:
                raise SystemExit(f"{case}: crop dropped class {need}")
            print(f"{case} class1={n1} class2={n2} RAS={np.round(ras, 2).tolist()}")
            cases.append(case)
    dataset = {
        "channel_names": {"0": "T1"},
        "labels": {"background": 0, "PC_left": 1, "PC_right": 2},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "licence": "Hammers n30 + Steinbart 2023 PC; 60 mm crop around gold PC",
        "dataset_name": "Dataset002_PiriformCrop60",
        "reference": "Steinbart 2023; crop analog Lyu 2023 / Mauri 2025 (inference FoV only in those papers)",
        "description": (
            "Stage A crop retrain. 60 mm per hemi around manual 125/126. "
            "Not JLF gold. Dataset001 whole-brain weights untouched."
        ),
    }
    (OUT / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    expand_splits()
    print(f"wrote {len(cases)} cases -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
